//! lints from https://rust-lang.github.io/rust-clippy/rust-1.82.0/index.html

enum LintGroup {
    // Cargo,
    // Complexity,
    // Correctness,
    Nursery,
    Pedantic,
    // Perf,
    // Restriction,
    // Style,
    // Suspicious,
}

enum LintApplicability {
    MachineApplicable,
    MaybeIncorrect,
    HasPlaceholders,
    Unspecified,
}

macro_rules! allow {
    (
        $lint:literal
        $(,group = $g:ident)?  $(,applicability = $a:ident)?
        $(,reason = $reason:literal)?
        $(,)?
    ) => {{
        $(let _ = LintGroup::$g;)*
        $(let _ = LintApplicability::$a;)*
        concat!("-A", "clippy::", $lint)
    }};
}

macro_rules! warn {
    (
        $lint:literal
        $(,group = $g:ident)?  $(,applicability = $a:ident)?
        $(,reason = $reason:literal)?
        $(,)?
    ) => {{
        $(let _ = LintGroup::$g;)*
        $(let _ = LintApplicability::$a;)*
        concat!("-W", "clippy::", $lint)
    }};
}

macro_rules! deny {
    (
        $lint:literal
        $(,group = $g:ident)?  $(,applicability = $a:ident)?
        $(,reason = $reason:literal)?
        $(,)?
    ) => {{
        $(let _ = LintGroup::$g;)*
        $(let _ = LintApplicability::$a;)*
        concat!("-D", "clippy::", $lint)
    }};
}

pub(crate) const CLIPPY_ARGS: &[&str] = &[
    deny!(
        "all",
        reason="Deny `all` default lints for better life! (correctness, suspicious, style, complexity, perf)"
    ),
    warn!(
        "pedantic",
        reason="Warning to all `pedantic`, then allow specific lints."
    ),

    allow!(
        "module_name_repetitions",
        group=Pedantic, applicability=Unspecified,
        reason="It is common to prefixed or suffixed by the containing module's name."
    ),

    // region:nursery
    deny!(
        "as_ptr_cast_mut",
        group=Nursery, applicability=MaybeIncorrect,
        reason="Avoid casting by `as`."
    ),
    deny!(
        "branches_sharing_code",
        group=Nursery, applicability=Unspecified,
        reason="Avoid duplicate code."
    ),
    deny!(
        "clear_with_drain",
        group=Nursery, applicability=MachineApplicable,
        reason="Cleanly express intention."
    ),
    deny!(
        "collection_is_never_read",
        group=Nursery, applicability=Unspecified,
        reason="Avoid unused variable."
    ),
    deny!(
        "debug_assert_with_mut_call",
        group=Nursery, applicability=Unspecified,
        reason="Avoid different behavior between a release and debug build."
    ),
    deny!(
        "derive_partial_eq_without_eq",
        group=Nursery, applicability=MachineApplicable,
        reason="Improve coherency"
    ),
    deny!(
        "empty_line_after_doc_comments",
        group=Nursery, applicability=Unspecified,
        reason="Format."
    ),
    deny!(
        "empty_line_after_outer_attr",
        group=Nursery, applicability=Unspecified,
        reason="Format."
    ),
    deny!(
        "equatable_if_let",
        group=Nursery, applicability=MachineApplicable,
        reason="Easy to understand."
    ),
    deny!(
        "fallible_impl_from",
        group=Nursery, applicability=Unspecified,
        reason="`TryFrom` should be used if there's a possibility of failure."
    ),
    deny!(
        "future_not_send",
        group=Nursery, applicability=Unspecified,
        reason="Future without send is hard to use."
    ),
    deny!(
        "imprecise_flops",
        group=Nursery, applicability=MachineApplicable,
        reason="Improve coherency."
    ),
    deny!(
        "iter_on_empty_collections",
        group=Nursery, applicability=MaybeIncorrect,
        reason="Improve coherency."
    ),
    deny!(
        "iter_with_drain",
        group=Nursery, applicability=MaybeIncorrect,
        reason="`.into_iter()` is simpler with better performance."
    ),
    deny!(
        "large_stack_frames",
        group=Nursery, applicability=Unspecified,
        reason="512000 Bytes stack object is rare, and should be avoid."
    ),
    deny!(
        "mutex_integer",
        group=Nursery, applicability=Unspecified,
        reason="`std::sync::atomic::AtomicUsize` is leaner and faster."
    ),
    deny!(
        "needless_collect",
        group=Nursery, applicability=MachineApplicable,
        reason="`collect` causes allocation."
    ),
    deny!(
        "needless_pass_by_ref_mut",
        group=Nursery, applicability=Unspecified,
        reason="Exclusive reference harms parallelization."
    ),
    deny!(
        "nonstandard_macro_braces",
        group=Nursery, applicability=MachineApplicable,
        reason="Follow standard."
    ),
    deny!(
        "path_buf_push_overwrite",
        group=Nursery, applicability=MachineApplicable,
        reason="Fix typo."
    ),
    deny!(
        "read_zero_byte_vec",
        group=Nursery, applicability=MaybeIncorrect,
        reason="Reading zero bytes is almost certainly not the intended behavior."
    ),
    deny!(
        "redundant_pub_crate",
        group=Nursery, applicability=MachineApplicable,
        reason="Remove unused attribute."
    ),
    deny!(
        "set_contains_or_insert",
        group=Nursery, applicability=Unspecified,
        reason="Using just insert and checking the returned bool is more efficient."
    ),
    deny!(
        "significant_drop_in_scrutinee",
        group=Nursery, applicability=MaybeIncorrect,
        reason="Avoid classic deadlock."
    ),
    deny!(
        "significant_drop_tightening",
        group=Nursery, applicability=MaybeIncorrect,
        reason="Drop early."
    ),
    deny!(
        "suspicious_operation_groupings",
        group=Nursery, applicability=MachineApplicable,
        reason="Fix typo."
    ),
    deny!(
        "trailing_empty_array",
        group=Nursery, applicability=Unspecified,
        reason="Ensure FFI safe."
    ),
    deny!(
        "trait_duplication_in_bounds",
        group=Nursery, applicability=MachineApplicable,
        reason="Avoid redundant trait bound."
    ),
    deny!(
        "transmute_undefined_repr",
        group=Nursery, applicability=Unspecified,
        reason="Avoid `transmute`."
    ),
    deny!(
        "trivial_regex",
        group=Nursery, applicability=Unspecified,
        reason="Prefer normal method."
    ),
    deny!(
        "type_repetition_in_bounds",
        group=Nursery, applicability=Unspecified,
        reason="Unnecessary type repetitions."
    ),
    deny!(
        "uninhabited_references",
        group=Nursery, applicability=Unspecified,
        reason="Avoid undefined behavior."
    ),
    deny!(
        "unnecessary_struct_initialization",
        group=Nursery, applicability=MachineApplicable,
        reason="Improve coherency."
    ),
    deny!(
        "unused_peekable",
        group=Nursery, applicability=Unspecified,
        reason="Avoid unused transition."
    ),
    deny!(
        "unused_rounding",
        group=Nursery, applicability=MachineApplicable,
        reason="Avoid unused transition."
    ),
    deny!(
        "useless_let_if_seq",
        group=Nursery, applicability=HasPlaceholders,
        reason="Writing idiomatic rust."
    ),

    warn!(
        "cognitive_complexity",
        group=Nursery, applicability=Unspecified,
        reason="Methods of high cognitive complexity tend to be hard to both read and maintain."
    ),
    warn!(
        "iter_on_single_items",
        group=Nursery, applicability=MaybeIncorrect,
        reason="Improve coherency, but it is common to use Some(T) to chain other iterator."
    ),
    warn!(
        "missing_const_for_fn",
        group=Nursery, applicability=MachineApplicable,
        reason="Always prefer const fn, but it is annoying."
    ),
    warn!(
        "option_if_let_else",
        group=Nursery, applicability=MaybeIncorrect,
        reason="Optimize to one line, may reduce readability."
    ),
    warn!(
        "or_fun_call",
        group=Nursery, applicability=HasPlaceholders,
        reason="`or` is more intuitive, but most of time we should use `or_else`."
    ),
    warn!(
        "redundant_clone",
        group=Nursery, applicability=MachineApplicable,
        reason="Too many false positive."
    ),
    warn!(
        "string_lit_as_bytes",
        group=Nursery, applicability=MachineApplicable,
        reason="Too many false positive."
    ),
    warn!(
        "suboptimal_flops",
        group=Nursery, applicability=MachineApplicable,
        reason="Improve coherency, but may have performance penalty."
    ),
    deny!(
        "use_self",
        group=Nursery, applicability=MachineApplicable,
        reason="Use `Self` to improve coherency, but may harms readability."
    ),

    allow!(
        "non_send_fields_in_send_ty",
        group=Nursery, applicability=Unspecified,
        reason="Impl Send Sync is often intended."
    ),
    allow!(
        "tuple_array_conversions",
        group=Nursery, applicability=Unspecified,
        reason="Hidden asymmetry."
    ),
    allow!(
        "while_float",
        group=Nursery, applicability=Unspecified,
        reason="Immature lint."
    ),

    // endregion:nursery

    // region:restriction

    // endregion:restriction
];
