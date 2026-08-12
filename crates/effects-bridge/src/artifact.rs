//! The metadata custom section: how a component artifact carries its own
//! effect signatures.
//!
//! A package is content-addressed over its whole artifact, so putting the
//! metadata inside the artifact is what makes a method's declared effects
//! and an index into its event table unable to drift from the code they
//! describe: change either and the address changes.
//!
//! The walk is section framing and nothing more — the same framing core
//! modules and components share — so extraction needs no engine, no
//! compilation, and no instantiation. Runtimes ignore a custom section
//! they do not know, which is what lets the artifact the chain stores be
//! the artifact the engine compiles.

use hyperscale_types::VmStaticsError;
use hyperscale_vm_effects::{
    AbiParam, Clause, MethodSignature, ModeExpr, PackageMetadata, TargetExpr, check_abi,
    check_declarations,
};
use hyperscale_vm_runtime::{ExportParam, component_export_params, validate_component};

use crate::vm_metadata::{decode_metadata, encode_metadata};

/// The custom section a component artifact declares its effect metadata
/// in.
pub const METADATA_SECTION: &str = "hyperscale:effect-metadata";

/// The section id wasm reserves for custom sections.
const CUSTOM_SECTION_ID: u8 = 0;

/// The magic and version word every module and component opens with.
const WASM_MAGIC: [u8; 4] = *b"\0asm";
const PREAMBLE_LEN: usize = 8;

/// Attach `metadata` to a component artifact as its metadata section.
///
/// The result is the publishable artifact: same code, one section longer,
/// and a different content address.
///
/// # Errors
///
/// [`VmStaticsError`] if the artifact's section framing is malformed, if
/// it already declares a metadata section, or if the metadata is past a
/// bound the codec enforces.
pub fn attach_metadata(
    artifact: &[u8],
    metadata: &PackageMetadata,
) -> Result<Vec<u8>, VmStaticsError> {
    if find_section(artifact)?.is_some() {
        return Err(VmStaticsError(
            "artifact already declares an effect metadata section".into(),
        ));
    }
    let payload = encode_metadata(metadata)?;

    let mut content = Vec::with_capacity(METADATA_SECTION.len() + payload.len() + 8);
    write_uleb128(METADATA_SECTION.len(), &mut content);
    content.extend_from_slice(METADATA_SECTION.as_bytes());
    content.extend_from_slice(&payload);

    let mut out = Vec::with_capacity(artifact.len() + content.len() + 8);
    out.extend_from_slice(artifact);
    out.push(CUSTOM_SECTION_ID);
    write_uleb128(content.len(), &mut out);
    out.extend_from_slice(&content);
    Ok(out)
}

/// The effect metadata a component artifact declares, if it declares any.
///
/// # Errors
///
/// [`VmStaticsError`] if the artifact's section framing is malformed, if
/// it declares the metadata section more than once, or if the section's
/// payload is not canonical metadata.
pub fn extract_metadata(artifact: &[u8]) -> Result<Option<PackageMetadata>, VmStaticsError> {
    find_section(artifact)?.map(decode_metadata).transpose()
}

/// The metadata section's payload, walking the artifact's sections.
///
/// Every step is checked against the bytes that remain, so a truncated
/// length, a section running past the artifact, or a name running past
/// its own section is a refusal rather than a panic. Two sections under
/// the name are refused as well: which one meant the package's effects
/// would otherwise be a question the format does not answer.
fn find_section(artifact: &[u8]) -> Result<Option<&[u8]>, VmStaticsError> {
    if artifact.len() < PREAMBLE_LEN || artifact[..WASM_MAGIC.len()] != WASM_MAGIC {
        return Err(VmStaticsError(
            "artifact does not open with the wasm preamble".into(),
        ));
    }
    let mut found: Option<&[u8]> = None;
    let mut pos = PREAMBLE_LEN;
    while pos < artifact.len() {
        let id = artifact[pos];
        pos += 1;
        let size = read_uleb128(artifact, &mut pos)?;
        let end = pos
            .checked_add(size)
            .filter(|end| *end <= artifact.len())
            .ok_or_else(|| VmStaticsError("section runs past the artifact".into()))?;

        if id == CUSTOM_SECTION_ID {
            // Bounded by the section's own end, so a name length cannot
            // read into whatever follows.
            let section = &artifact[..end];
            let mut inner = pos;
            let name_len = read_uleb128(section, &mut inner)?;
            let name_end = inner
                .checked_add(name_len)
                .filter(|name_end| *name_end <= end)
                .ok_or_else(|| {
                    VmStaticsError("custom section name runs past its section".into())
                })?;
            if &artifact[inner..name_end] == METADATA_SECTION.as_bytes() {
                if found.is_some() {
                    return Err(VmStaticsError(
                        "artifact declares the effect metadata section twice".into(),
                    ));
                }
                found = Some(&artifact[name_end..end]);
            }
        }
        pos = end;
    }
    Ok(found)
}

/// The metadata a publish admits from an artifact, or why it does not.
///
/// Five things are checkable today, and they are checked: the artifact
/// clears the deterministic profile, it declares a metadata section at
/// all, the section decodes canonically and within the bounds the
/// vocabulary fixes, every method it describes is a function the
/// component actually exports, and each method's ABI binding agrees with
/// that export's own type — same arity, a capability handle where the
/// export takes a borrow of the resource the clause's mode implies, a
/// bucket's amount where it takes bytes. Whether a signature
/// over-approximates the code it describes is a compiler's judgement,
/// and this is not one — an under-declaration is harmless because the
/// capability gate never materialises a handle the declaration did not
/// ask for, so a wrong signature costs its author a trap rather than
/// costing anyone else safety. A binding that disagrees with the export
/// is different: the disagreement surfaces at invocation, through
/// whatever error channel each runtime happens to have, so it is refused
/// here where the verdict is one.
///
/// Every one of these is a pure function of the artifact's bytes, which
/// is what lets the whole verdict be reached at admission rather than
/// split across admission and execution. A publish that cannot be
/// admitted never enters a block, so nobody pays for it and nobody
/// stores it.
///
/// # Errors
///
/// [`VmStaticsError`] on an artifact outside the profile, an absent or
/// non-canonical metadata section, a declared method the component does
/// not export, or an ABI binding the export's type cannot honour.
pub fn admit_package(artifact: &[u8]) -> Result<PackageMetadata, VmStaticsError> {
    validate_component(artifact)
        .map_err(|error| VmStaticsError(format!("artifact is outside the profile: {error}")))?;
    let metadata = extract_metadata(artifact)?
        .ok_or_else(|| VmStaticsError("artifact declares no effect metadata section".into()))?;
    let exports = component_export_params(artifact)
        .map_err(|error| VmStaticsError(format!("artifact does not parse: {error}")))?;
    for (method, signature) in &metadata.methods {
        let Some(params) = exports.get(method.as_str()) else {
            return Err(VmStaticsError(format!(
                "metadata declares method {method:?}, which the component does not export"
            )));
        };
        // The binding is the vocabulary's to judge: every clause is a
        // pure function of the signature, and routing judges the same
        // predicate again for a package that reached a cache without
        // ever passing this gate.
        check_abi(signature)
            .map_err(|error| VmStaticsError(format!("method {method:?}: {error}")))?;
        check_declarations(signature)
            .map_err(|error| VmStaticsError(format!("method {method:?}: {error}")))?;
        check_abi_against_export(method, signature, params)?;
    }
    Ok(metadata)
}

/// Judge a method's ABI binding against the export type that will
/// receive the arguments it builds.
///
/// `check_abi` has already judged the binding against the signature, so
/// clause and parameter indices resolve; what remains is whether the
/// compiled export can take what the binding builds.
fn check_abi_against_export(
    method: &str,
    signature: &MethodSignature,
    params: &[ExportParam],
) -> Result<(), VmStaticsError> {
    if signature.abi.len() != params.len() {
        return Err(VmStaticsError(format!(
            "method {method:?}: the binding builds {} arguments, the export takes {}",
            signature.abi.len(),
            params.len()
        )));
    }
    for (position, (binding, param)) in signature.abi.iter().zip(params).enumerate() {
        match binding {
            AbiParam::Handle(clause) => {
                let ExportParam::Handle(resource) = param else {
                    return Err(VmStaticsError(format!(
                        "method {method:?}: ABI parameter {position} is a capability \
                         handle, but the export takes {param:?}"
                    )));
                };
                let expected = usize::try_from(*clause)
                    .ok()
                    .and_then(|index| signature.effects.get(index))
                    .and_then(expected_resource);
                if let Some(expected) = expected
                    && resource != expected
                {
                    return Err(VmStaticsError(format!(
                        "method {method:?}: ABI parameter {position} borrows \
                         `{resource}`, but clause {clause}'s mode materialises a \
                         `{expected}`"
                    )));
                }
            }
            AbiParam::Bucket(_) => {
                if *param != ExportParam::Bytes {
                    return Err(VmStaticsError(format!(
                        "method {method:?}: ABI parameter {position} is a bucket \
                         amount, but the export takes {param:?}"
                    )));
                }
            }
            AbiParam::Derived(_) => {
                if matches!(param, ExportParam::Handle(_)) {
                    return Err(VmStaticsError(format!(
                        "method {method:?}: ABI parameter {position} is a derived \
                         value, but the export takes a resource borrow"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// The state resource a clause's handle parameter borrows, when the
/// clause pins one statically.
///
/// A `for-each` clause yields `None`: naming one as a handle parameter
/// is a deterministic refusal at materialization, so there is no single
/// resource to hold the export to here.
const fn expected_resource(clause: &Clause) -> Option<&'static str> {
    let Clause::Effect { target, mode } = clause else {
        return None;
    };
    match (target, mode) {
        (TargetExpr::Point(_), ModeExpr::Read) => Some("read-cell"),
        (TargetExpr::Point(_), ModeExpr::Locked) => Some("locked-cell"),
        (TargetExpr::Point(_), ModeExpr::Write) => Some("write-cell"),
        (TargetExpr::Point(_), ModeExpr::Delta) => Some("delta-cell"),
        (TargetExpr::Point(_), ModeExpr::Reserve(_)) => Some("reserve-cell"),
        (TargetExpr::Entry { .. } | TargetExpr::Range { .. }, ModeExpr::Read) => Some("range-read"),
        (TargetExpr::Entry { .. } | TargetExpr::Range { .. }, ModeExpr::Write) => {
            Some("range-write")
        }
        _ => None,
    }
}

fn write_uleb128(mut value: usize, out: &mut Vec<u8>) {
    loop {
        let byte = u8::try_from(value & 0x7F).expect("seven bits fit a byte");
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Read one wasm `u32` length, capped at the five bytes the encoding
/// admits so a padded run cannot spin.
fn read_uleb128(bytes: &[u8], pos: &mut usize) -> Result<usize, VmStaticsError> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = *bytes
            .get(*pos)
            .ok_or_else(|| VmStaticsError("section length is truncated".into()))?;
        *pos += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 32 {
            return Err(VmStaticsError(
                "section length is not a 32-bit value".into(),
            ));
        }
    }
    if value > u64::from(u32::MAX) {
        return Err(VmStaticsError(
            "section length is not a 32-bit value".into(),
        ));
    }
    usize::try_from(value)
        .map_err(|_| VmStaticsError("section length does not fit this platform".into()))
}

#[cfg(test)]
mod tests {

    use hyperscale_vm_effects::stdlib::{account_metadata, book_metadata};
    use hyperscale_vm_effects::{AbiParam, Accessibility, Expr, MethodSignature, PackageMetadata};
    use hyperscale_vm_stdlib::{account_artifact, staking_artifact};
    use wat::parse_str;

    use super::*;

    /// A component exporting one no-argument function per name.
    fn component_exporting(names: &[&str]) -> Vec<u8> {
        use std::fmt::Write as _;

        let mut source = String::from("(component\n  (core module $m\n");
        for index in 0..names.len() {
            let _ = writeln!(source, "    (func (export \"f{index}\"))");
        }
        source.push_str("  )\n  (core instance $i (instantiate $m))\n");
        for (index, name) in names.iter().enumerate() {
            let _ = writeln!(
                source,
                "  (func (export \"{name}\") (canon lift (core func $i \"f{index}\")))"
            );
        }
        source.push(')');
        parse_str(&source).expect("the component assembles")
    }

    /// Metadata declaring one empty signature per method name.
    fn declaring(methods: &[&str]) -> PackageMetadata {
        let mut metadata = PackageMetadata::default();
        for method in methods {
            metadata
                .methods
                .insert((*method).into(), MethodSignature::default());
        }
        metadata
    }

    /// The smallest well-formed artifact shape the walk accepts: a
    /// preamble and nothing else.
    fn bare() -> Vec<u8> {
        let mut out = WASM_MAGIC.to_vec();
        out.extend_from_slice(&[0x0d, 0x00, 0x01, 0x00]);
        out
    }

    /// A preamble followed by one non-custom section carrying `body`.
    fn with_section(id: u8, body: &[u8]) -> Vec<u8> {
        let mut out = bare();
        out.push(id);
        write_uleb128(body.len(), &mut out);
        out.extend_from_slice(body);
        out
    }

    /// A binding a caller cannot resolve is refused where it is cheapest
    /// to refuse: the artifact's own bytes, before a block carries it.
    ///
    /// The predicate itself is the vocabulary's and is tested there. What
    /// this pins is that a publish consults it at all, and that its
    /// refusal names the method whose binding is wrong — a package
    /// declaring several is otherwise unactionable.
    #[test]
    fn an_unresolvable_abi_binding_refuses_at_publish() {
        let component = component_exporting(&["m"]);
        let mut metadata = declaring(&["m"]);
        metadata
            .methods
            .get_mut("m")
            .expect("declared")
            // The signature declares no effect clauses, so there is no
            // clause 0 for a handle to name.
            .abi = vec![AbiParam::Handle(0)];
        let artifact = attach_metadata(&component, &metadata).expect("attaches");

        let refused = admit_package(&artifact).expect_err("an unresolvable binding refuses");
        assert!(refused.0.contains("\"m\""), "{}", refused.0);

        // The same artifact with nothing bound admits, so the refusal is
        // the binding and not the shape.
        let sound = attach_metadata(&component, &declaring(&["m"])).expect("attaches");
        assert!(admit_package(&sound).is_ok());
    }

    #[test]
    fn an_artifact_declares_the_metadata_it_was_attached() {
        for metadata in [account_metadata(), book_metadata()] {
            let plain = with_section(1, b"code goes here");
            assert_eq!(extract_metadata(&plain).expect("walks"), None);

            let artifact = attach_metadata(&plain, &metadata).expect("attaches");
            assert_eq!(
                extract_metadata(&artifact).expect("walks"),
                Some(metadata.clone())
            );
            // The code is untouched and the artifact is a different one.
            assert!(artifact.starts_with(&plain));
            assert_ne!(artifact, plain);
        }
    }

    #[test]
    fn different_metadata_makes_a_different_artifact() {
        // What content addressing over the whole artifact buys: the
        // declared effects cannot drift from the code under one address.
        let plain = with_section(1, b"code");
        let one = attach_metadata(&plain, &account_metadata()).expect("attaches");
        let other = attach_metadata(&plain, &book_metadata()).expect("attaches");
        assert_ne!(one, other);
    }

    #[test]
    fn the_section_is_found_past_other_custom_sections() {
        // A real component carries name and producers sections; the walk
        // has to skip custom sections it does not know, and must not
        // match on a prefix of the name either.
        let mut plain = with_section(1, b"code");
        for name in ["name", "producers", "hyperscale:effect-metadata-x"] {
            let mut content = Vec::new();
            write_uleb128(name.len(), &mut content);
            content.extend_from_slice(name.as_bytes());
            content.extend_from_slice(b"payload");
            plain.push(CUSTOM_SECTION_ID);
            write_uleb128(content.len(), &mut plain);
            plain.extend_from_slice(&content);
        }
        assert_eq!(extract_metadata(&plain).expect("walks"), None);

        let artifact = attach_metadata(&plain, &account_metadata()).expect("attaches");
        assert_eq!(
            extract_metadata(&artifact).expect("walks"),
            Some(account_metadata())
        );
    }

    #[test]
    fn a_second_metadata_section_is_refused() {
        let artifact =
            attach_metadata(&with_section(1, b"code"), &account_metadata()).expect("attaches");
        // Attaching again is refused rather than producing an artifact
        // whose metadata is ambiguous.
        assert!(attach_metadata(&artifact, &book_metadata()).is_err());

        // And an artifact assembled with two anyway does not extract.
        let mut doubled = artifact.clone();
        doubled.extend_from_slice(&artifact[with_section(1, b"code").len()..]);
        assert!(extract_metadata(&doubled).is_err());
    }

    #[test]
    fn malformed_framing_is_refused_rather_than_walked() {
        let artifact =
            attach_metadata(&with_section(1, b"code"), &account_metadata()).expect("attaches");

        // No preamble at all, and a preamble that is not wasm's.
        assert!(extract_metadata(b"").is_err());
        assert!(extract_metadata(&artifact[..4]).is_err());
        assert!(extract_metadata(&[0u8; 16]).is_err());

        // A section claiming more bytes than the artifact holds.
        let mut overrun = bare();
        overrun.push(1);
        write_uleb128(64, &mut overrun);
        overrun.extend_from_slice(b"short");
        assert!(extract_metadata(&overrun).is_err());

        // A length that never terminates, and one padded past 32 bits.
        let mut truncated = bare();
        truncated.extend_from_slice(&[1, 0x80]);
        assert!(extract_metadata(&truncated).is_err());
        let mut oversized = bare();
        oversized.extend_from_slice(&[1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00]);
        assert!(extract_metadata(&oversized).is_err());

        // A custom section with no room for its own name.
        let mut nameless = bare();
        nameless.push(CUSTOM_SECTION_ID);
        write_uleb128(0, &mut nameless);
        assert!(extract_metadata(&nameless).is_err());

        // A name longer than the section that carries it.
        let mut overlong = bare();
        let mut content = Vec::new();
        write_uleb128(64, &mut content);
        content.extend_from_slice(b"name");
        overlong.push(CUSTOM_SECTION_ID);
        write_uleb128(content.len(), &mut overlong);
        overlong.extend_from_slice(&content);
        assert!(extract_metadata(&overlong).is_err());

        // Truncating the payload leaves the framing intact and the
        // metadata undecodable, which is a refusal and not a None.
        let mut clipped = artifact.clone();
        clipped.truncate(artifact.len() - 1);
        assert!(extract_metadata(&clipped).is_err());
    }

    #[test]
    fn a_corrupt_payload_is_refused_rather_than_read() {
        let plain = with_section(1, b"code");
        let artifact = attach_metadata(&plain, &account_metadata()).expect("attaches");
        // Every byte the section's payload occupies: a change either
        // fails to decode or names different metadata, never silently
        // the same.
        for index in plain.len()..artifact.len() {
            let mut mutated = artifact.clone();
            mutated[index] ^= 0xFF;
            if let Ok(Some(metadata)) = extract_metadata(&mutated) {
                assert_ne!(metadata, account_metadata());
            }
        }
    }

    #[test]
    fn a_publish_admits_metadata_the_component_backs() {
        let component = component_exporting(&["deposit", "withdraw"]);
        let metadata = declaring(&["deposit", "withdraw"]);
        let artifact = attach_metadata(&component, &metadata).expect("attaches");
        assert_eq!(admit_package(&artifact).expect("admits"), metadata);

        // Declaring fewer methods than the component exports is fine:
        // an export nothing declares is an export nothing can call.
        let partial = attach_metadata(&component, &declaring(&["deposit"])).expect("attaches");
        assert!(admit_package(&partial).is_ok());
    }

    /// What a publish admits is what the package declared, down to who
    /// may call each method.
    ///
    /// The gate at admission reads this field and nothing else to decide
    /// whether a node needs its target's signature, so a codec that
    /// dropped it would not fail loudly — it would publish every method
    /// as public and leave the gate agreeing.
    #[test]
    fn a_publish_admits_the_accessibility_the_package_declares() {
        let component = component_exporting(&["deposit", "withdraw"]);
        let mut metadata = declaring(&["deposit", "withdraw"]);
        metadata
            .methods
            .get_mut("withdraw")
            .expect("declared")
            .accessibility = Accessibility::Guarded(Expr::SelfAddr);
        let artifact = attach_metadata(&component, &metadata).expect("attaches");

        let admitted = admit_package(&artifact).expect("admits");
        assert_eq!(
            admitted.methods["withdraw"].accessibility,
            Accessibility::Guarded(Expr::SelfAddr)
        );
        assert_eq!(
            admitted.methods["deposit"].accessibility,
            Accessibility::Public
        );

        // And the two declarations are two artifacts: the field is
        // content-addressed with the code, so nothing can republish the
        // same address under a weaker claim.
        let public =
            attach_metadata(&component, &declaring(&["deposit", "withdraw"])).expect("attaches");
        assert_ne!(artifact, public);
    }

    #[test]
    fn a_publish_refuses_a_method_the_component_does_not_export() {
        let component = component_exporting(&["deposit"]);
        let artifact =
            attach_metadata(&component, &declaring(&["deposit", "withdraw"])).expect("attaches");
        let refused = admit_package(&artifact).expect_err("refuses");
        assert!(refused.0.contains("withdraw"), "{}", refused.0);

        // The name has to match exactly — a component export is looked
        // up by the name a manifest node writes.
        let renamed = attach_metadata(
            &component_exporting(&["deposit2"]),
            &declaring(&["deposit"]),
        )
        .expect("attaches");
        assert!(admit_package(&renamed).is_err());
    }

    #[test]
    fn a_publish_refuses_an_artifact_that_declares_nothing() {
        // No signatures, no deploy: an artifact without the section is
        // refused rather than published with an empty table.
        let component = component_exporting(&["deposit"]);
        assert!(admit_package(&component).is_err());
        // And one whose section is not parseable as an artifact at all.
        assert!(admit_package(&with_section(1, b"code")).is_err());
    }

    #[test]
    fn only_the_outermost_components_exports_count() {
        // A nested component's exports are its own; nothing a manifest
        // names can reach them, so they cannot back a declaration.
        let inner = "(component (core module $m (func (export \"f\"))) \
             (core instance $i (instantiate $m)) \
             (func (export \"hidden\") (canon lift (core func $i \"f\"))))";
        let outer = parse_str(&*format!(
            "(component (core module $m (func (export \"f\"))) \
             (core instance $i (instantiate $m)) \
             (func (export \"shown\") (canon lift (core func $i \"f\"))) \
             {inner})"
        ))
        .expect("the component assembles");

        let exports = component_export_params(&outer).expect("parses");
        assert_eq!(exports.keys().collect::<Vec<_>>(), vec!["shown"]);
        let artifact = attach_metadata(&outer, &declaring(&["hidden"])).expect("attaches");
        assert!(admit_package(&artifact).is_err());
    }

    /// The committed stdlib artifacts pass the same gate a runtime
    /// publish would: their authored metadata agrees with the export
    /// types their blobs compile to. Without this the stdlib's binding
    /// is judged by nothing — genesis seeds it into the cache directly.
    #[test]
    fn the_stdlib_artifacts_pass_the_publish_gate() {
        for (name, artifact) in [
            ("account", account_artifact()),
            ("staking", staking_artifact()),
        ] {
            admit_package(artifact)
                .unwrap_or_else(|error| panic!("{name}: the stdlib must admit: {}", error.0));
        }
    }

    /// A component whose one export takes a `u64`, for bindings to
    /// disagree with.
    fn scalar_export() -> Vec<u8> {
        parse_str(
            r#"(component
                 (core module $m
                   (func (export "f") (param i64) (result i64) local.get 0))
                 (core instance $i (instantiate $m))
                 (func (export "m") (param "clock" u64) (result u64)
                   (canon lift (core func $i "f"))))"#,
        )
        .expect("the component assembles")
    }

    #[test]
    fn a_binding_the_export_type_cannot_honour_refuses_at_publish() {
        use hyperscale_vm_effects::{Clause, Expr, ModeExpr, RoleId, TargetExpr};

        // Arity: the binding builds nothing, the export takes one.
        let empty = declaring(&["m"]);
        let artifact = attach_metadata(&scalar_export(), &empty).expect("attaches");
        let refused = admit_package(&artifact).expect_err("arity must refuse");
        assert!(refused.0.contains("arguments"), "{}", refused.0);

        // A handle binding against a scalar parameter.
        let mut wrong_kind = declaring(&["m"]);
        {
            let signature = wrong_kind.methods.get_mut("m").expect("declared");
            signature.effects = vec![Clause::Effect {
                target: TargetExpr::Point(Expr::ChildKey {
                    owner: Box::new(Expr::SelfAddr),
                    role: RoleId(1),
                    material: vec![],
                }),
                mode: ModeExpr::Write,
            }];
            signature.abi = vec![AbiParam::Handle(0)];
        }
        let artifact = attach_metadata(&scalar_export(), &wrong_kind).expect("attaches");
        let refused = admit_package(&artifact).expect_err("a handle needs a borrow");
        assert!(refused.0.contains("capability handle"), "{}", refused.0);

        // A derived value where the export takes a borrow of the wrong
        // resource for the clause's mode.
        let borrow_export = parse_str(
            r#"(component
                 (import "hyperscale:kernel/state" (instance $state
                   (export "delta-cell" (type $dc (sub resource)))))
                 (alias export $state "delta-cell" (type $delta))
                 (core module $m
                   (func (export "f") (param i32) (result i64) i64.const 0))
                 (core instance $i (instantiate $m))
                 (func (export "m") (param "vault" (borrow $delta)) (result u64)
                   (canon lift (core func $i "f"))))"#,
        )
        .expect("the component assembles");
        let mut wrong_resource = declaring(&["m"]);
        {
            let signature = wrong_resource.methods.get_mut("m").expect("declared");
            signature.effects = vec![Clause::Effect {
                target: TargetExpr::Point(Expr::ChildKey {
                    owner: Box::new(Expr::SelfAddr),
                    role: RoleId(1),
                    material: vec![],
                }),
                mode: ModeExpr::Reserve(Expr::Arg(0)),
            }];
            signature.abi = vec![AbiParam::Handle(0)];
        }
        let artifact = attach_metadata(&borrow_export, &wrong_resource).expect("attaches");
        let refused = admit_package(&artifact).expect_err("the borrowed resource must match");
        assert!(refused.0.contains("reserve-cell"), "{}", refused.0);

        // The same shape with the matching mode admits, so the refusals
        // above are the disagreement and not the shape.
        let mut sound = declaring(&["m"]);
        {
            let signature = sound.methods.get_mut("m").expect("declared");
            signature.effects = vec![Clause::Effect {
                target: TargetExpr::Point(Expr::ChildKey {
                    owner: Box::new(Expr::SelfAddr),
                    role: RoleId(1),
                    material: vec![],
                }),
                mode: ModeExpr::Delta,
            }];
            signature.abi = vec![AbiParam::Handle(0)];
        }
        let artifact = attach_metadata(&borrow_export, &sound).expect("attaches");
        assert!(admit_package(&artifact).is_ok());
    }
}
