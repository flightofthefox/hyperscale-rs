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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Condvar, Mutex};

use arc_swap::ArcSwap;
use hyperscale_vm_effects::PackageHash;

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

/// The build verdict for guest code by content address, growable while
/// invocations run.
///
/// Shared between the target-gated backends — each fills it with its own
/// compiled form — so package resolution cannot drift between targets
/// the way a per-module map could. The settled map takes the metadata
/// cache's shape: lock-free loads on the invoke path, clone-and-swap on
/// the rare publish, first write wins by content address. `None` records
/// a build that refused these bytes, which is as settled an answer as a
/// build that landed and is what keeps a refusal from being re-fetched
/// and rebuilt forever. The pending set is what keeps cache state out of
/// verdicts: an invocation arriving while its package compiles waits the
/// work out rather than answering differently than a replica whose
/// compile already finished.
struct PackageSlots<C> {
    settled: ArcSwap<BTreeMap<PackageHash, Option<Arc<C>>>>,
    pending: Mutex<BTreeSet<PackageHash>>,
    done: Condvar,
}

impl<C> PackageSlots<C> {
    fn new() -> Self {
        Self {
            settled: ArcSwap::from_pointee(BTreeMap::new()),
            pending: Mutex::new(BTreeSet::new()),
            done: Condvar::new(),
        }
    }

    /// The runnable form of `package`, waiting out an in-flight compile.
    ///
    /// `None` when the package was never absorbed or its build refused it
    /// — both deterministic functions of committed bytes, so every
    /// replica answers alike.
    fn resolve(&self, package: PackageHash) -> Option<Arc<C>> {
        if let Some(verdict) = self.settled.load().get(&package) {
            return verdict.clone();
        }
        let mut pending = self.pending.lock().expect("package slots lock poisoned");
        while pending.contains(&package) {
            pending = self
                .done
                .wait(pending)
                .expect("package slots lock poisoned");
        }
        drop(pending);
        self.settled.load().get(&package).cloned().flatten()
    }

    /// Claim the build of `package`: `true` exactly once per content
    /// address, `false` when its verdict is settled or in flight.
    fn claim(&self, package: PackageHash) -> bool {
        if self.settled.load().contains_key(&package) {
            return false;
        }
        self.pending
            .lock()
            .expect("package slots lock poisoned")
            .insert(package)
    }

    /// Land a claimed build's verdict — `None` a refusal — and release
    /// its waiters.
    ///
    /// One producer only: the swap over `settled` is a read-modify-write
    /// with no lock around it, so two builds landing at once would lose
    /// one. Every claim reaches exactly one builder — the compile worker
    /// on the blessed engine, the absorbing thread on the reference one —
    /// and construction lands the stdlib before either exists.
    fn fulfil(&self, package: PackageHash, code: Option<C>) {
        let mut next = (**self.settled.load()).clone();
        next.entry(package).or_insert_with(|| code.map(Arc::new));
        self.settled.store(Arc::new(next));
        let mut pending = self.pending.lock().expect("package slots lock poisoned");
        pending.remove(&package);
        drop(pending);
        self.done.notify_all();
    }

    /// Whether `package` resolves without waiting, in-flight work
    /// excluded. A refused build qualifies: it resolves to no code, and
    /// it resolves there on every replica.
    fn is_settled(&self, package: PackageHash) -> bool {
        self.settled.load().contains_key(&package)
    }

    /// Whether `package`'s verdict is settled or its build is in flight
    /// — the probe that keeps a prefetch from re-requesting bytes the
    /// backend has already judged.
    fn is_known(&self, package: PackageHash) -> bool {
        if self.is_settled(package) {
            return true;
        }
        self.pending
            .lock()
            .expect("package slots lock poisoned")
            .contains(&package)
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;
    use std::thread;

    use crossbeam::channel::{Sender, unbounded};
    use hyperscale_effects_bridge::ProtocolHasher;
    use hyperscale_vm_effects::{PackageHash, package_hash};
    use hyperscale_vm_kernel::{
        AbortReason, CellKind, GuestArg, GuestBackend as Backend, GuestCall, InvokeResult,
        KernelSession,
    };
    use hyperscale_vm_runtime::{
        CellKind as HostCellKind, HostArg, add_kernel_to_linker, blessed_engine, call_export,
        classify, exhausted as fuel_exhausted, validate_component,
    };
    use wasmtime::component::{Component, InstancePre, Linker};
    use wasmtime::{Engine, Store};

    use super::{FUEL, HostState, PackageSlots};
    use crate::genesis::GenesisPackages;

    /// The compiled guests, pre-linked for cheap instantiation.
    pub struct EngineBackend {
        engine: Engine,
        /// Compiled code by content address. A lowered call names the
        /// package it runs, never the instance, because code is what a
        /// backend resolves and code is what a content address covers —
        /// so two instances of one package share one compilation and two
        /// packages in one transaction each get their own.
        slots: Arc<PackageSlots<InstancePre<HostState>>>,
        /// Feed of the compile worker: a dedicated OS thread, never the
        /// shared dispatch pools — wasmtime's internal parallel
        /// compilation nested inside a pooled worker is a known
        /// self-deadlock shape. The thread exits when the last sender
        /// drops.
        compile: Sender<Vec<u8>>,
    }

    impl EngineBackend {
        /// Compile `packages` on the blessed engine and start the
        /// compile worker for everything published after them.
        ///
        /// The artifact compiled is the one the package address covers,
        /// metadata section included: what the chain stores is what the
        /// engine runs. The set is the network's genesis set, because a
        /// package the chain is born holding is one no node ever fetches
        /// — every node compiles it at boot instead.
        ///
        /// # Panics
        ///
        /// Panics if a genesis artifact fails profile validation or
        /// compilation — a build defect, not a runtime condition.
        pub fn new(packages: &GenesisPackages) -> Self {
            let engine = blessed_engine().expect("blessed engine configuration is pinned");
            let linker = kernel_linker(&engine);
            let slots = Arc::new(PackageSlots::new());
            for artifact in packages.artifacts() {
                validate_component(artifact).expect("a genesis artifact clears the profile");
                let pre = build(&engine, &linker, artifact).expect("a genesis artifact compiles");
                let package = package_hash(&ProtocolHasher, artifact);
                assert!(slots.claim(package), "genesis packages are distinct");
                slots.fulfil(package, Some(pre));
            }

            let (compile_tx, compile_rx) = unbounded::<Vec<u8>>();
            let worker_engine = engine.clone();
            let worker_slots = Arc::clone(&slots);
            thread::Builder::new()
                .name("package-compile".into())
                .spawn(move || {
                    let linker = kernel_linker(&worker_engine);
                    for artifact in compile_rx {
                        let package = package_hash(&ProtocolHasher, &artifact);
                        worker_slots.fulfil(package, build(&worker_engine, &linker, &artifact));
                    }
                })
                .expect("the compile worker spawns");

            Self {
                engine,
                slots,
                compile: compile_tx,
            }
        }

        /// Queue a committed package's artifact for compilation.
        ///
        /// Idempotent by content address; the compiled code becomes
        /// resolvable when the worker lands it, and an invocation
        /// arriving sooner waits on exactly that.
        pub fn absorb_artifact(&self, artifact: &[u8]) {
            queue(&self.slots, &self.compile, artifact);
        }

        /// A cheap-clone feed of this backend for the commit path to
        /// hold: [`Self::absorb_artifact`] detached from the borrow.
        pub fn absorber(&self) -> impl Fn(&[u8]) + Send + Sync + 'static {
            let slots = Arc::clone(&self.slots);
            let compile = self.compile.clone();
            move |artifact: &[u8]| queue(&slots, &compile, artifact)
        }

        /// Whether `package`'s code resolves without waiting — landed or
        /// refused.
        #[must_use]
        pub fn code_settled(&self, package: PackageHash) -> bool {
            self.slots.is_settled(package)
        }

        /// Whether `package`'s code is judged or being built.
        #[must_use]
        pub fn code_known(&self, package: PackageHash) -> bool {
            self.slots.is_known(package)
        }
    }

    /// A linker carrying the kernel world — the one import surface a
    /// deployable component may name.
    fn kernel_linker(engine: &Engine) -> Linker<HostState> {
        let mut linker = Linker::<HostState>::new(engine);
        add_kernel_to_linker(&mut linker).expect("kernel world wiring");
        linker
    }

    /// Claim `artifact`'s build and hand it to the compile worker.
    ///
    /// A claim and a send, in that order, so two callers racing the same
    /// bytes queue one build. A send that fails means the worker is gone
    /// — the claim then stands unfulfilled and every call to the package
    /// waits on a build that will never land, which is a dead node
    /// rather than a divergent one. Loud, because nothing downstream can
    /// tell that apart from a slow fetch.
    fn queue(
        slots: &PackageSlots<InstancePre<HostState>>,
        compile: &Sender<Vec<u8>>,
        artifact: &[u8],
    ) {
        let package = package_hash(&ProtocolHasher, artifact);
        if slots.claim(package) && compile.send(artifact.to_vec()).is_err() {
            tracing::error!(
                ?package,
                "the compile worker is gone; its packages cannot run"
            );
        }
    }

    /// Build one artifact, or `None` if the blessed engine refuses it.
    ///
    /// Compilation is one pinned wasmtime over one blessed config, so
    /// every way it can end is a function of the bytes and every replica
    /// reaches the same one — an unwind included, which is why the
    /// worker catches rather than dies on it. What is not deterministic
    /// is the profile validator having admitted bytes wasmtime will not
    /// take, so a refusal here is logged as the disagreement it is.
    fn build(
        engine: &Engine,
        linker: &Linker<HostState>,
        artifact: &[u8],
    ) -> Option<InstancePre<HostState>> {
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            let component =
                Component::new(engine, artifact).map_err(|error| format!("compile: {error:#}"))?;
            linker
                .instantiate_pre(&component)
                .map_err(|error| format!("link: {error:#}"))
        }));
        let reason = match attempt {
            Ok(Ok(pre)) => return Some(pre),
            Ok(Err(reason)) => reason,
            Err(_) => "panic".to_string(),
        };
        tracing::error!(
            package = ?package_hash(&ProtocolHasher, artifact),
            reason,
            "published artifact failed to compile"
        );
        None
    }

    impl Backend for EngineBackend {
        fn invoke(&self, session: KernelSession, call: &GuestCall<'_>) -> InvokeResult {
            // What the transaction has left, under the per-invocation
            // ceiling: a manifest's nodes draw from one signed budget.
            let budget = call.fuel_budget.min(FUEL);
            let mut store = Store::new(&self.engine, HostState(session));
            store.set_fuel(budget).expect("fuel metering is enabled");
            let Some(pre) = self.slots.resolve(call.package) else {
                return InvokeResult {
                    session: store.into_data().0,
                    fuel: 0,
                    result: Err(AbortReason::CodeUnavailable),
                    exhausted: false,
                };
            };
            let instance = match pre.instantiate(&mut store) {
                Ok(instance) => instance,
                Err(error) => {
                    tracing::debug!(?error, "component did not instantiate");
                    return InvokeResult {
                        session: store.into_data().0,
                        fuel: 0,
                        result: Err(AbortReason::InstantiationFailed),
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
            let exhausted = outcome.as_ref().err().is_some_and(fuel_exhausted);
            let result = outcome.map_err(|error| {
                let reason = classify(&error);
                tracing::debug!(export = call.export, ?reason, detail = ?error, "guest aborted");
                reason
            });
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
    use std::sync::Arc;

    use hyperscale_effects_bridge::ProtocolHasher;
    use hyperscale_vm_effects::{PackageHash, package_hash};
    use hyperscale_vm_kernel::{
        AbortReason, CellKind, GuestArg, GuestBackend as Backend, GuestCall, InvokeResult,
        KernelSession,
    };
    use hyperscale_vm_ref::{
        CVal, ExecError, RefComponent, RefComponentInstance, ResourceKind, Trap as RefTrap,
    };
    use hyperscale_vm_runtime::validate_component;

    use super::{FUEL, HostState, PackageSlots};
    use crate::genesis::GenesisPackages;

    /// The decoded guests under the reference interpreter.
    pub struct EngineBackend {
        slots: Arc<PackageSlots<RefComponent>>,
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
        /// Panics if a genesis artifact fails profile validation or
        /// decoding — a build defect, not a runtime condition.
        pub fn new(packages: &GenesisPackages) -> Self {
            let slots = Arc::new(PackageSlots::new());
            for artifact in packages.artifacts() {
                validate_component(artifact).expect("a genesis artifact clears the profile");
                let component = RefComponent::decode(artifact).expect("a genesis artifact decodes");
                let package = package_hash(&ProtocolHasher, artifact);
                assert!(slots.claim(package), "genesis packages are distinct");
                slots.fulfil(package, Some(component));
            }
            Self { slots }
        }

        /// Absorb a committed package's artifact.
        ///
        /// Decoding is one parser pass, so this target does it in place
        /// — no worker, and the pending set never holds an entry long
        /// enough for an invocation to wait on it.
        pub fn absorb_artifact(&self, artifact: &[u8]) {
            absorb_into(&self.slots, artifact);
        }

        /// A cheap-clone feed of this backend for the commit path to
        /// hold: [`Self::absorb_artifact`] detached from the borrow.
        pub fn absorber(&self) -> impl Fn(&[u8]) + Send + Sync + 'static {
            let slots = Arc::clone(&self.slots);
            move |artifact: &[u8]| absorb_into(&slots, artifact)
        }

        /// Whether `package`'s code resolves without waiting — landed or
        /// refused.
        #[must_use]
        pub fn code_settled(&self, package: PackageHash) -> bool {
            self.slots.is_settled(package)
        }

        /// Whether `package`'s code is judged or being built.
        #[must_use]
        pub fn code_known(&self, package: PackageHash) -> bool {
            self.slots.is_known(package)
        }
    }

    fn absorb_into(slots: &PackageSlots<RefComponent>, artifact: &[u8]) {
        let package = package_hash(&ProtocolHasher, artifact);
        if !slots.claim(package) {
            return;
        }
        match RefComponent::decode(artifact) {
            Ok(component) => slots.fulfil(package, Some(component)),
            Err(error) => {
                tracing::error!(?package, %error, "published artifact failed to decode");
                slots.fulfil(package, None);
            }
        }
    }

    impl Backend for EngineBackend {
        fn invoke(&self, session: KernelSession, call: &GuestCall<'_>) -> InvokeResult {
            let args: Vec<CVal> = call.args.iter().map(ref_arg).collect();
            let Some(component) = self.slots.resolve(call.package) else {
                return InvokeResult {
                    session,
                    fuel: 0,
                    result: Err(AbortReason::CodeUnavailable),
                    exhausted: false,
                };
            };
            let mut instance =
                match RefComponentInstance::instantiate(&component, HostState(session)) {
                    Ok(instance) => instance,
                    Err((host, error)) => {
                        tracing::debug!(?error, "component did not instantiate");
                        return InvokeResult {
                            session: host.0,
                            fuel: 0,
                            result: Err(AbortReason::InstantiationFailed),
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
                    other => {
                        tracing::debug!(export = call.export, ?other, "off-convention result");
                        Err(AbortReason::BadReturnShape)
                    }
                },
                Ok(Err(error)) => {
                    let reason = error.abort_reason();
                    tracing::debug!(export = call.export, ?reason, ?error, "guest aborted");
                    Err(reason)
                }
                // The export is not in the component's table, which the
                // publish gate admitted it against.
                Err(error) => {
                    tracing::debug!(export = call.export, ?error, "export not invocable");
                    Err(AbortReason::ExportMissing)
                }
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
