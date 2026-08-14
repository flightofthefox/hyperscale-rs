//! The package-metadata section codec's chain-policy byte budget.
//!
//! The codec itself is the vocabulary's own — canonical HBOR of
//! [`PackageMetadata`] at the vocabulary's wire depth, judged by its own
//! bounds. What belongs here is the one thing the vocabulary deliberately
//! does not fix: the byte budget a section may claim of a transaction.

use hyperscale_types::{MAX_TX_BYTES_LEN, VmStaticsError};
use hyperscale_vm_effects::{
    PackageMetadata, decode_metadata as decode_canonical, encode_metadata as encode_canonical,
};

/// The bound on an encoded metadata section.
///
/// A section rides inside a published artifact and the artifact inside a
/// transaction, so the code it describes has to fit beside it; a quarter
/// of the transaction budget is the share this side claims. The cap is
/// also what makes decode linear: HBOR frames every collection with its
/// length and every element costs at least a byte, so no claimed count
/// can outrun the input.
pub const MAX_PACKAGE_METADATA_BYTES: usize = MAX_TX_BYTES_LEN / 4;

/// Encode package metadata into its canonical section bytes.
///
/// # Errors
///
/// [`VmStaticsError`] if the metadata is past a bound decode enforces, so
/// that whatever this returns decodes back to an equal value.
pub fn encode_metadata(metadata: &PackageMetadata) -> Result<Vec<u8>, VmStaticsError> {
    let bytes = encode_canonical(metadata).map_err(|error| VmStaticsError(error.to_string()))?;
    if bytes.len() > MAX_PACKAGE_METADATA_BYTES {
        return Err(VmStaticsError(format!(
            "metadata encodes to {} bytes, past the {MAX_PACKAGE_METADATA_BYTES} cap",
            bytes.len()
        )));
    }
    Ok(bytes)
}

/// Decode a metadata section's canonical bytes.
///
/// # Errors
///
/// [`VmStaticsError`] on an oversized section, malformed or non-canonical
/// bytes, or a structure past a bound the vocabulary fixes.
pub fn decode_metadata(bytes: &[u8]) -> Result<PackageMetadata, VmStaticsError> {
    if bytes.len() > MAX_PACKAGE_METADATA_BYTES {
        return Err(VmStaticsError(format!(
            "metadata section is {} bytes, past the {MAX_PACKAGE_METADATA_BYTES} cap",
            bytes.len()
        )));
    }
    decode_canonical(bytes).map_err(|error| VmStaticsError(error.to_string()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use hyperscale_hbor::to_vec_with_depth;
    use hyperscale_types::MAX_EVENT_TYPES;
    use hyperscale_vm_effects::stdlib::{
        VAULT, account_metadata, amm_metadata, book_metadata, splitter_metadata,
    };
    use hyperscale_vm_effects::{
        Accessibility, Address, AddressClass, CallSite, Clause, EdgeContent, Expr, LocalKey,
        MAX_CLAUSE_DEPTH, MAX_EFFECTS_PER_SIGNATURE, MAX_EXPR_DEPTH, MAX_VALUE_DEPTH,
        METADATA_WIRE_DEPTH, MethodSignature, ModeExpr, ParamType, RoleId, SubstateKey, TargetExpr,
        Value,
    };

    use super::*;

    fn stdlib() -> Vec<(&'static str, PackageMetadata)> {
        vec![
            ("account", account_metadata()),
            ("amm", amm_metadata()),
            ("book", book_metadata()),
            ("splitter", splitter_metadata()),
        ]
    }

    /// A signature whose only effect points at `expr`.
    fn signature_over(expr: Expr) -> MethodSignature {
        MethodSignature {
            effects: vec![Clause::Effect {
                target: TargetExpr::Point(expr),
                mode: ModeExpr::Write,
            }],
            ..MethodSignature::default()
        }
    }

    fn one_method(signature: MethodSignature) -> PackageMetadata {
        let mut metadata = PackageMetadata::default();
        metadata.methods.insert("m".into(), signature);
        metadata
    }

    /// The section bytes for metadata the bound checks would refuse —
    /// what a hostile publisher writes, and the only input that puts the
    /// decode-side bounds under test.
    fn encode_unchecked(metadata: &PackageMetadata) -> Vec<u8> {
        to_vec_with_depth(metadata, METADATA_WIRE_DEPTH)
            .expect("the vocabulary encodes within the codec's own nesting cap")
    }

    /// Both sides of a bound: the admitted structure round trips, and the
    /// one past it is refused by encode and by decode alike.
    fn assert_bounded(admitted: &PackageMetadata, refused: &PackageMetadata) {
        let bytes = encode_metadata(admitted).expect("the admitted structure encodes");
        assert_eq!(&decode_metadata(&bytes).expect("decodes"), admitted);
        assert!(
            encode_metadata(refused).is_err(),
            "encode accepted a structure past the bound"
        );
        assert!(
            decode_metadata(&encode_unchecked(refused)).is_err(),
            "decode accepted a structure past the bound"
        );
    }

    /// A left-nested projection chain, the shape the evaluator's own depth
    /// test uses.
    fn nested_projection(depth: usize) -> Expr {
        let mut expr = Expr::Arg(0);
        for _ in 0..depth {
            expr = Expr::Field(Box::new(expr), 0);
        }
        expr
    }

    fn nested_foreach(depth: usize) -> Clause {
        let mut clause = Clause::Effect {
            target: TargetExpr::Point(Expr::SelfAddr),
            mode: ModeExpr::Read,
        };
        for _ in 0..depth {
            clause = Clause::ForEach {
                list: Expr::Arg(0),
                body: vec![clause],
            };
        }
        clause
    }

    #[test]
    fn the_stdlib_metadata_round_trips() {
        for (package, metadata) in stdlib() {
            let bytes = encode_metadata(&metadata).expect("encodes");
            let decoded = decode_metadata(&bytes).expect("decodes");
            assert_eq!(decoded, metadata, "{package} metadata round trip");
            assert_eq!(
                encode_metadata(&decoded).expect("re-encodes"),
                bytes,
                "{package} metadata re-encodes identically"
            );
        }
    }

    #[test]
    fn every_authored_shape_survives_the_codec() {
        // The stdlib does not author every variant, so the coverage the
        // round-trip test cannot give comes from one method that does:
        // each expression form, each target form, each mode, a nested
        // for-each body, a call site, and a deep literal.
        let signature = MethodSignature {
            accessibility: Accessibility::Guarded(Expr::SelfAddr),
            mints: None,
            abi: Vec::new(),
            params: vec![
                ParamType::U64,
                ParamType::U128,
                ParamType::Bytes,
                ParamType::Address,
                ParamType::Bucket,
            ],
            outputs: vec![
                Expr::Config(2),
                Expr::ResourceOf(Box::new(Expr::Arg(4))),
                Expr::Literal(Value::Tuple(vec![
                    Value::U64(1),
                    Value::List(vec![Value::Bytes(vec![7, 8, 9])]),
                    Value::Key(SubstateKey {
                        owner: Address::new([3; 31], AddressClass::Component),
                        local: LocalKey([4; 16]),
                    }),
                    Value::Bucket {
                        content: EdgeContent::Fungible,
                        resource: Address::new([5; 31], AddressClass::Component),
                    },
                    Value::U128(u128::MAX),
                    Value::Address(Address::new([6; 31], AddressClass::Component)),
                ])),
            ],
            effects: vec![
                Clause::Effect {
                    target: TargetExpr::Point(Expr::ChildKey {
                        owner: Box::new(Expr::SelfAddr),
                        role: VAULT,
                        material: vec![Expr::Arg(3), Expr::FreshKey { slot: 1 }],
                    }),
                    mode: ModeExpr::Reserve(Expr::Arg(1)),
                },
                Clause::Effect {
                    target: TargetExpr::Entry {
                        owner: Expr::Field(Box::new(Expr::Config(0)), 2),
                        collection: RoleId(9),
                        material: vec![],
                        order: Expr::Pack {
                            hi: Box::new(Expr::Arg(0)),
                            lo: Box::new(Expr::FreshId { slot: 3 }),
                        },
                    },
                    mode: ModeExpr::Locked,
                },
                Clause::Effect {
                    target: TargetExpr::Range {
                        owner: Expr::SelfAddr,
                        collection: RoleId(4),
                        material: vec![],
                        lo: Expr::Literal(Value::U128(0)),
                        hi: Expr::Literal(Value::U128(u128::MAX)),
                        cap: 64,
                    },
                    mode: ModeExpr::Locked,
                },
                Clause::ForEach {
                    list: Expr::Arg(2),
                    body: vec![Clause::ForEach {
                        list: Expr::Binding(0),
                        body: vec![Clause::Effect {
                            target: TargetExpr::Point(Expr::Lookup {
                                map: Box::new(Expr::Binding(1)),
                                key: Box::new(Expr::Binding(0)),
                            }),
                            mode: ModeExpr::Delta,
                        }],
                    }],
                },
                Clause::Effect {
                    target: TargetExpr::Point(Expr::SelfAddr),
                    mode: ModeExpr::Read,
                },
            ],
            calls: vec![CallSite {
                target: Expr::Config(1),
                method: "deposit".into(),
                args: vec![Expr::Arg(4), Expr::Literal(Value::U64(11))],
            }],
        };
        let mut metadata = one_method(signature);
        metadata
            .methods
            .insert("another".into(), MethodSignature::default());
        metadata.events = vec!["withdrawn".into(), "deposited".into()];

        let bytes = encode_metadata(&metadata).expect("encodes");
        assert_eq!(decode_metadata(&bytes).expect("decodes"), metadata);
    }

    #[test]
    fn any_byte_change_fails_or_changes_the_value() {
        for (package, metadata) in stdlib() {
            let bytes = encode_metadata(&metadata).expect("encodes");
            for index in 0..bytes.len() {
                for mask in [0x01u8, 0x40, 0x80, 0xFF] {
                    let mut mutated = bytes.clone();
                    mutated[index] ^= mask;
                    if let Ok(decoded) = decode_metadata(&mutated) {
                        assert_ne!(
                            decoded, metadata,
                            "{package}: byte {index} ^ {mask:#04x} decoded to the same value"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn truncated_and_extended_payloads_are_refused() {
        let bytes = encode_metadata(&account_metadata()).expect("encodes");
        for cut in 0..bytes.len() {
            assert!(
                decode_metadata(&bytes[..cut]).is_err(),
                "a payload truncated to {cut} bytes decoded"
            );
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert!(
            decode_metadata(&trailing).is_err(),
            "trailing byte accepted"
        );
        assert!(decode_metadata(&[]).is_err());
        assert!(decode_metadata(&[0xFF, 0x00]).is_err());
    }

    #[test]
    fn the_method_table_decodes_only_in_ascending_name_order() {
        // The table travels sorted, so permuting it or repeating a name
        // is a distinct byte string that must not decode to a value the
        // map would silently normalise. A sequence of pairs encodes
        // byte-identically to a map whose entries arrive in the same
        // order, so this is how a forged table is spelled at all — the
        // map form cannot hold one.
        #[derive(hyperscale_hbor::Hbor)]
        struct Forged {
            methods: Vec<(String, MethodSignature)>,
            events: Vec<String>,
        }

        let mut metadata = PackageMetadata::default();
        for name in ["a", "b"] {
            metadata
                .methods
                .insert(name.into(), MethodSignature::default());
        }
        let bytes = encode_metadata(&metadata).expect("encodes");
        let rewrite = |names: &[&str]| {
            to_vec_with_depth(
                &Forged {
                    methods: names
                        .iter()
                        .map(|name| ((*name).to_owned(), MethodSignature::default()))
                        .collect(),
                    events: Vec::new(),
                },
                METADATA_WIRE_DEPTH,
            )
            .expect("the forged table encodes")
        };
        assert_eq!(rewrite(&["a", "b"]), bytes);
        assert!(decode_metadata(&rewrite(&["b", "a"])).is_err());
        assert!(decode_metadata(&rewrite(&["a", "a"])).is_err());
    }

    #[test]
    fn the_deepest_admissible_metadata_still_encodes() {
        // Every nesting bound at its limit at once, along the costliest
        // path: clause bodies, child-key material, and a tuple literal
        // each cost two decoder levels a layer. If the codec's own nesting
        // limit ever stops covering the bounds it is derived from, this
        // is what says so — the checks would accept a structure the
        // encoder could not write.
        let mut literal = Value::U64(0);
        for _ in 1..MAX_VALUE_DEPTH {
            literal = Value::Tuple(vec![literal]);
        }
        let mut deepest = Expr::Literal(literal);
        for _ in 0..MAX_EXPR_DEPTH {
            deepest = Expr::ChildKey {
                owner: Box::new(Expr::SelfAddr),
                role: VAULT,
                material: vec![deepest],
            };
        }
        let mut clause = Clause::Effect {
            target: TargetExpr::Range {
                owner: Expr::SelfAddr,
                collection: RoleId(4),
                material: vec![],
                lo: deepest.clone(),
                hi: deepest,
                cap: 1,
            },
            mode: ModeExpr::Locked,
        };
        for _ in 0..MAX_CLAUSE_DEPTH {
            clause = Clause::ForEach {
                list: Expr::Arg(0),
                body: vec![clause],
            };
        }
        let metadata = one_method(MethodSignature {
            effects: vec![clause],
            ..MethodSignature::default()
        });

        let bytes = encode_metadata(&metadata).expect("the deepest admissible metadata encodes");
        assert_eq!(decode_metadata(&bytes).expect("decodes"), metadata);
    }

    #[test]
    fn expression_nesting_is_bounded_where_the_evaluator_bounds_it() {
        assert_bounded(
            &one_method(signature_over(nested_projection(MAX_EXPR_DEPTH))),
            &one_method(signature_over(nested_projection(MAX_EXPR_DEPTH + 1))),
        );
    }

    #[test]
    fn clause_nesting_is_bounded_where_the_evaluator_bounds_it() {
        assert_bounded(
            &one_method(MethodSignature {
                effects: vec![nested_foreach(MAX_CLAUSE_DEPTH)],
                ..MethodSignature::default()
            }),
            &one_method(MethodSignature {
                effects: vec![nested_foreach(MAX_CLAUSE_DEPTH + 1)],
                ..MethodSignature::default()
            }),
        );
    }

    #[test]
    fn a_clause_tree_wider_than_a_signature_can_declare_is_refused() {
        let effect = Clause::Effect {
            target: TargetExpr::Point(Expr::SelfAddr),
            mode: ModeExpr::Read,
        };
        let with = |count: usize| {
            one_method(MethodSignature {
                effects: vec![effect.clone(); count],
                ..MethodSignature::default()
            })
        };
        assert_bounded(
            &with(MAX_EFFECTS_PER_SIGNATURE),
            &with(MAX_EFFECTS_PER_SIGNATURE + 1),
        );
    }

    #[test]
    fn literal_nesting_is_bounded_where_admission_bounds_it() {
        let literal = |depth: usize| {
            let mut value = Value::U64(0);
            for _ in 1..depth {
                value = Value::Tuple(vec![value]);
            }
            Expr::Literal(value)
        };
        assert_bounded(
            &one_method(signature_over(literal(MAX_VALUE_DEPTH))),
            &one_method(signature_over(literal(MAX_VALUE_DEPTH + 1))),
        );
    }

    #[test]
    fn an_event_table_past_the_index_the_kernel_accepts_is_refused() {
        let table = |len: usize| PackageMetadata {
            methods: BTreeMap::new(),
            events: vec![String::new(); len],
        };
        assert_bounded(
            &table(MAX_EVENT_TYPES as usize),
            &table(MAX_EVENT_TYPES as usize + 1),
        );
    }

    #[test]
    fn an_oversized_section_is_refused_before_it_is_parsed() {
        // Well formed but past the cap: an event table spending more
        // than the section budget, refused on length before the decoder
        // reads a byte of it.
        let over = PackageMetadata {
            methods: BTreeMap::new(),
            events: vec!["e".repeat(1024); MAX_EVENT_TYPES as usize],
        };
        let bytes = encode_unchecked(&over);
        assert!(bytes.len() > MAX_PACKAGE_METADATA_BYTES);
        assert!(decode_metadata(&bytes).is_err());
        assert!(encode_metadata(&over).is_err());

        assert!(decode_metadata(&vec![0u8; MAX_PACKAGE_METADATA_BYTES + 1]).is_err());
    }
}
