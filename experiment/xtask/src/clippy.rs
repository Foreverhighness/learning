//! lints from <https://rust-lang.github.io/rust-clippy/rust-1.82.0/index.html>
#![allow(dead_code, reason = "allow dead_code")]

enum LintGroup {
    Cargo,
    Complexity,
    Correctness,
    Nursery,
    Pedantic,
    Perf,
    Restriction,
    Style,
    Suspicious,
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

pub const CLIPPY_ARGS: &[&str] = &[
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
        "or_fun_call",
        group=Nursery, applicability=HasPlaceholders,
        reason="`or` is more intuitive, but most of time we should use `or_else`."
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
    allow!(
        "option_if_let_else",
        group=Nursery, applicability=MaybeIncorrect,
        reason="Greatly reduce readability."
    ),
    allow!(
        "redundant_clone",
        group=Nursery, applicability=MachineApplicable,
        reason="Too many false positive."
    ),
    allow!(
        "string_lit_as_bytes",
        group=Nursery, applicability=MachineApplicable,
        reason="Too many false positive."
    ),

    // endregion:nursery

    // region:restriction

    // deny!(
    //     "alloc_instead_of_core",
    //     group=Restriction, applicability=MachineApplicable,
    //     reason="Writing `no_std` friendly code."
    // ),
    // deny!(
    //     "std_instead_of_core",
    //     group=Restriction, applicability=MachineApplicable,
    //     reason="Writing `no_std` friendly code."
    // ),
    // deny!(
    //     "std_instead_of_alloc",
    //     group=Restriction, applicability=Unspecified,
    //     reason="Writing `no_std` friendly code."
    // ),

    deny!(
        "allow_attributes",
        group=Restriction, applicability=MachineApplicable,
        reason="Prefer use `expect` attribute than `allow`."
    ),
    deny!(
        "allow_attributes_without_reason",
        group=Restriction, applicability=Unspecified,
        reason="Allow attribute should explain the reason."
    ),
    deny!(
        "clone_on_ref_ptr",
        group=Restriction, applicability=Unspecified,
        reason="Prefer explicit clone over cheap clone types."
    ),
    deny!(
        "default_union_representation",
        group=Restriction, applicability=Unspecified,
        reason="`union` should be `repr(C)`."
    ),
    deny!(
        "empty_drop",
        group=Restriction, applicability=MaybeIncorrect,
        reason="Disallow empty `Drop` implement."
    ),
    deny!(
        "error_impl_error",
        group=Restriction, applicability=Unspecified,
        reason="Errors should implement `Error` trait."
    ),
    deny!(
        "filetype_is_file",
        group=Restriction, applicability=Unspecified,
        reason="`is_file` doesn't cover special file types in unix-like systems, and doesn't cover symlink in windows."
    ),
    deny!(
        "format_push_string",
        group=Restriction, applicability=Unspecified,
        reason="Avoid heap allocation."
    ),
    deny!(
        "if_then_some_else_none",
        group=Restriction, applicability=Unspecified,
        reason="More concise and incurs no loss of clarity."
    ),
    deny!(
        "infinite_loop",
        group=Restriction, applicability=MaybeIncorrect,
        reason="Explicit mark function that is never return."
    ),
    deny!(
        "lossy_float_literal",
        group=Restriction, applicability=MachineApplicable,
        reason="Be careful about precision."
    ),
    deny!(
        "mem_forget",
        group=Restriction, applicability=Unspecified,
        reason="Prefer `ManuallyDrop` instead of `mem::forget`."
    ),
    deny!(
        "mixed_read_write_in_expression",
        group=Restriction, applicability=Unspecified,
        reason="Do not use confusing syntax."
    ),
    deny!(
        "modulo_arithmetic",
        group=Restriction, applicability=Unspecified,
        reason="Modulo with negative number is hard to understand."
    ),
    deny!(
        "multiple_unsafe_ops_per_block",
        group=Restriction, applicability=Unspecified,
        reason="Reduce unsafe block granularity."
    ),
    deny!(
        "pattern_type_mismatch",
        group=Restriction, applicability=Unspecified,
        reason="My favorite lint."
    ),
    deny!(
        "rc_buffer",
        group=Restriction, applicability=Unspecified,
        reason="`Rc` is readonly."
    ),
    deny!(
        "rc_mutex",
        group=Restriction, applicability=Unspecified,
        reason="`Rc<Mutex<T>>` is so wired."
    ),
    deny!(
        "same_name_method",
        group=Restriction, applicability=Unspecified,
        reason="Avoid same method name, which is confusing."
    ),
    deny!(
        "str_to_string",
        group=Restriction, applicability=MachineApplicable,
        reason="`ToOwned` is more specific."
    ),
    deny!(
        "string_add",
        group=Restriction, applicability=Unspecified,
        reason="Prefer `String::push_str` than `+=`."
    ),
    deny!(
        "string_to_string",
        group=Restriction, applicability=Unspecified,
        reason="Correct semantics."
    ),
    deny!(
        "try_err",
        group=Restriction, applicability=MachineApplicable,
        reason="Explicit return `Err`."
    ),
    deny!(
        "unneeded_field_pattern",
        group=Restriction, applicability=Unspecified,
        reason="More concise."
    ),
    deny!(
        "unused_result_ok",
        group=Restriction, applicability=MaybeIncorrect,
        reason="Avoid unnecessary"
    ),
    deny!(
        "verbose_file_reads",
        group=Restriction, applicability=Unspecified,
        reason="More concise."
    ),

    warn!(
        "decimal_literal_representation",
        group=Restriction, applicability=MaybeIncorrect,
        reason="Hexadecimal representation sometimes is more readable than a decimal representation."
    ),
    warn!(
        "undocumented_unsafe_blocks",
        group=Restriction, applicability=Unspecified,
        reason="Improve readability."
    ),

    // endregion:restriction
];
