//! The guest invocation backend: the blessed engine on native targets,
//! the reference interpreter on wasm32.
//!
//! One instantiation per guest call — the execution model fuel parity is
//! pinned against — with the session threaded in and out through the
//! host state. Traps come back as deterministic reason strings; the
//! session always survives for the kernel's rollback.
//!
//! The seam is an engine seam and nothing more: the kernel hands over an
//! export name and an argument list it assembled from the transaction's
//! own declaration, so an embedder here can get engine embedding wrong
//! and cannot get manifest semantics wrong.

use crate::host::HostState;

/// The ceiling on what one invocation may consume, whatever its
/// transaction declared.
///
/// Consensus content: exhaustion is a deterministic trap, so the two
/// engines have to meter against one number. It lives outside both
/// backend modules because they are target-gated and never compile
/// together — a per-module constant could drift between targets with
/// nothing to catch it, and the divergence would only surface as two
/// nodes disagreeing on whether a runaway guest trapped.
const FUEL: u64 = 10_000_000;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::BTreeMap;

    use hyperscale_effects_bridge::ProtocolHasher;
    use hyperscale_vm_effects::{PackageHash, package_hash};
    use hyperscale_vm_kernel::{
        CellKind, GuestArg, GuestBackend as Backend, GuestCall, InvokeResult, KernelSession,
    };
    use hyperscale_vm_runtime::{
        CellKind as HostCellKind, HostArg, add_kernel_to_linker, blessed_engine, call_export,
        validate_component,
    };
    use wasmtime::component::{Component, InstancePre, Linker};
    use wasmtime::{Engine, Store, Trap};

    use super::{FUEL, HostState};
    use crate::genesis::{account_artifact, staking_artifact};

    /// The compiled account guest, pre-linked for cheap instantiation.
    pub struct EngineBackend {
        engine: Engine,
        /// Compiled code by content address. A lowered call names the
        /// package it runs, never the instance, because code is what a
        /// backend resolves and code is what a content address covers —
        /// so two instances of one package share one compilation and two
        /// packages in one transaction each get their own.
        packages: BTreeMap<PackageHash, InstancePre<HostState>>,
    }

    impl EngineBackend {
        /// Compile the genesis packages on the blessed engine.
        ///
        /// The artifact compiled is the one the package address covers,
        /// metadata section included: what the chain stores is what the
        /// engine runs.
        ///
        /// # Panics
        ///
        /// Panics if the stdlib artifact fails profile validation or
        /// compilation — a build defect, not a runtime condition.
        pub fn new() -> Self {
            let engine = blessed_engine().expect("blessed engine configuration is pinned");
            let mut linker = Linker::<HostState>::new(&engine);
            add_kernel_to_linker(&mut linker).expect("kernel world wiring");
            let mut packages = BTreeMap::new();
            for artifact in [account_artifact(), staking_artifact()] {
                validate_component(artifact).expect("a stdlib artifact clears the profile");
                let component =
                    Component::new(&engine, artifact).expect("a stdlib artifact compiles");
                packages.insert(
                    package_hash(&ProtocolHasher, artifact),
                    linker
                        .instantiate_pre(&component)
                        .expect("a stdlib component links against the kernel world"),
                );
            }
            Self { engine, packages }
        }
    }

    impl Backend for EngineBackend {
        fn invoke(&self, session: KernelSession, call: &GuestCall<'_>) -> InvokeResult {
            // What the transaction has left, under the per-invocation
            // ceiling: a manifest's nodes draw from one signed budget.
            let budget = call.fuel_budget.min(FUEL);
            let mut store = Store::new(&self.engine, HostState(session));
            store.set_fuel(budget).expect("fuel metering is enabled");
            let Some(pre) = self.packages.get(&call.package) else {
                return InvokeResult {
                    session: store.into_data().0,
                    fuel: 0,
                    result: Err("no compiled code for the called package".to_string()),
                    exhausted: false,
                };
            };
            let instance = match pre.instantiate(&mut store) {
                Ok(instance) => instance,
                Err(error) => {
                    return InvokeResult {
                        session: store.into_data().0,
                        fuel: 0,
                        result: Err(format!("instantiate: {error:#}")),
                        exhausted: false,
                    };
                }
            };
            let args: Vec<HostArg<'_>> = call.args.iter().map(host_arg).collect();
            let outcome = call_export(&mut store, &instance, call.export, &args, call.returns);
            // The engine's own classification, not an inference from the
            // fuel left: a trap that happens to land on an exhausted
            // counter is a different outcome from one caused by it, and
            // the reference interpreter reports the distinction exactly.
            // Two runtimes disagreeing here is two nodes disagreeing on
            // whether a transaction was its sender's own defect.
            let exhausted = outcome
                .as_ref()
                .err()
                .and_then(|error| error.downcast_ref::<Trap>())
                .is_some_and(|trap| matches!(trap, Trap::OutOfFuel));
            let result = outcome.map_err(|trap| format!("{trap:#}"));
            let remaining = store.get_fuel().expect("fuel metering is enabled");
            let fuel = budget - remaining;
            InvokeResult {
                session: store.into_data().0,
                fuel,
                result,
                exhausted,
            }
        }
    }

    const fn host_kind(kind: CellKind) -> HostCellKind {
        match kind {
            CellKind::Read => HostCellKind::Read,
            CellKind::Locked => HostCellKind::Locked,
            CellKind::Write => HostCellKind::Write,
            CellKind::Delta => HostCellKind::Delta,
            CellKind::Reserve => HostCellKind::Reserve,
            CellKind::RangeRead => HostCellKind::RangeRead,
            CellKind::RangeWrite => HostCellKind::RangeWrite,
        }
    }

    const fn host_arg<'a>(arg: &GuestArg<'a>) -> HostArg<'a> {
        match arg {
            GuestArg::Handle { rep, kind } => HostArg::Handle {
                rep: *rep,
                kind: host_kind(*kind),
            },
            GuestArg::U64(scalar) => HostArg::U64(*scalar),
            GuestArg::Bytes(bytes) => HostArg::Bytes(bytes),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::EngineBackend;

#[cfg(target_arch = "wasm32")]
mod reference {
    use std::collections::BTreeMap;

    use hyperscale_effects_bridge::ProtocolHasher;
    use hyperscale_vm_effects::{PackageHash, package_hash};
    use hyperscale_vm_kernel::{
        CellKind, GuestArg, GuestBackend as Backend, GuestCall, InvokeResult, KernelSession,
    };
    use hyperscale_vm_ref::{
        CVal, ExecError, RefComponent, RefComponentInstance, ResourceKind, Trap as RefTrap,
    };
    use hyperscale_vm_runtime::validate_component;

    use super::{FUEL, HostState};
    use crate::genesis::{account_artifact, staking_artifact};

    /// The decoded stdlib guests under the reference interpreter.
    pub struct EngineBackend {
        packages: BTreeMap<PackageHash, RefComponent>,
    }

    impl EngineBackend {
        /// Decode the genesis packages.
        ///
        /// The artifact clears the same profile it clears under the
        /// blessed engine: the verdict is a property of the bytes, and
        /// a build that interprets components rather than compiling them
        /// has no less need of it.
        ///
        /// # Panics
        ///
        /// Panics if the stdlib artifact fails profile validation or
        /// decoding — a build defect, not a runtime condition.
        pub fn new() -> Self {
            let mut packages = BTreeMap::new();
            for artifact in [account_artifact(), staking_artifact()] {
                validate_component(artifact).expect("a stdlib artifact clears the profile");
                packages.insert(
                    package_hash(&ProtocolHasher, artifact),
                    RefComponent::decode(artifact).expect("a stdlib artifact decodes"),
                );
            }
            Self { packages }
        }
    }

    impl Backend for EngineBackend {
        fn invoke(&self, session: KernelSession, call: &GuestCall<'_>) -> InvokeResult {
            let args: Vec<CVal> = call.args.iter().map(ref_arg).collect();
            let Some(component) = self.packages.get(&call.package) else {
                return InvokeResult {
                    session,
                    fuel: 0,
                    result: Err("no decoded code for the called package".to_string()),
                    exhausted: false,
                };
            };
            let mut instance =
                match RefComponentInstance::instantiate(component, HostState(session)) {
                    Ok(instance) => instance,
                    Err((host, error)) => {
                        return InvokeResult {
                            session: host.0,
                            fuel: 0,
                            result: Err(format!("instantiate: {error}")),
                            exhausted: false,
                        };
                    }
                };
            // The same ceiling the blessed engine meters against. Without
            // it this engine is unbounded, and a guest past the budget
            // traps on one target and runs on the other.
            instance.set_fuel_limit(call.fuel_budget.min(FUEL));
            let outcome = instance.invoke(call.export, &args);
            let exhausted = matches!(outcome, Ok(Err(ExecError::Trap(RefTrap::OutOfFuel))));
            let fuel = instance.fuel_consumed();
            let session = instance.into_host().0;
            let result = match outcome {
                Ok(Ok(values)) => match (call.returns, values.as_slice()) {
                    (false, []) => Ok(None),
                    (true, [CVal::Bytes(bytes)]) => Ok(Some(bytes.clone())),
                    other => Err(format!("unexpected result shape: {other:?}")),
                },
                Ok(Err(trap)) => Err(format!("{trap:?}")),
                Err(error) => Err(format!("invoke: {error:?}")),
            };
            InvokeResult {
                session,
                fuel,
                result,
                exhausted,
            }
        }
    }

    const fn ref_kind(kind: CellKind) -> ResourceKind {
        match kind {
            CellKind::Read => ResourceKind::ReadCell,
            CellKind::Locked => ResourceKind::LockedCell,
            CellKind::Write => ResourceKind::WriteCell,
            CellKind::Delta => ResourceKind::DeltaCell,
            CellKind::Reserve => ResourceKind::ReserveCell,
            CellKind::RangeRead => ResourceKind::RangeRead,
            CellKind::RangeWrite => ResourceKind::RangeWrite,
        }
    }

    fn ref_arg(arg: &GuestArg<'_>) -> CVal {
        match arg {
            GuestArg::Handle { rep, kind } => CVal::Borrow(*rep, ref_kind(*kind)),
            GuestArg::U64(scalar) => CVal::U64(*scalar),
            GuestArg::Bytes(bytes) => CVal::Bytes(bytes.to_vec()),
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use reference::EngineBackend;
