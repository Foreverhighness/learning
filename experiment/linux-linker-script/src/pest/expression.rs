use std::sync::LazyLock;

use derive_more::derive::{Display, From};
use pest::iterators::Pair;
use pest::pratt_parser::{Assoc, Op, PrattParser};

use super::ast::AbstractSyntaxTreeNode;
use super::commands::Assertion;
use super::parser::Rule;

#[derive(Debug)]
pub struct Expression {
    unary_op: String,
    primary: Primary,
    exprs: Vec<(Operation, String, Primary)>,
}

impl Expression {
    const fn new(primary: Primary) -> Self {
        Self {
            unary_op: String::new(),
            primary,
            exprs: Vec::new(),
        }
    }

    fn with_op(self, op: Operation) -> Vec<(Operation, String, Primary)> {
        let Self {
            unary_op,
            primary,
            mut exprs,
        } = self;

        exprs.insert(0, (op, unary_op, primary));
        exprs
    }
}

static EXPR_PRATT_PARSER: LazyLock<PrattParser<Rule>> = LazyLock::new(|| {
    // Precedence is defined lowest to highest
    // https://en.cppreference.com/w/c/language/operator_precedence
    PrattParser::new()
        .op(Op::infix(Rule::logical_or, Assoc::Left)) // 12
        .op(Op::infix(Rule::bitwise_and, Assoc::Left)) // 8
        .op(Op::infix(Rule::eq, Assoc::Left)) // 7
        .op(Op::infix(Rule::le, Assoc::Left)) // 6
        .op(Op::infix(Rule::left_shift, Assoc::Left) | Op::infix(Rule::right_shift, Assoc::Left)) // 5
        .op(Op::infix(Rule::add, Assoc::Left) | Op::infix(Rule::sub, Assoc::Left)) // 4
        .op(Op::infix(Rule::mul, Assoc::Left)) // 3
        .op(Op::prefix(Rule::neg) | Op::prefix(Rule::bitwise_not)) // 2
});

impl AbstractSyntaxTreeNode for Expression {
    fn parse(pair: Pair<'_, Rule>) -> Self {
        assert!(matches!(pair.as_rule(), Rule::expr));

        EXPR_PRATT_PARSER
            .map_primary(|primary| Self::new(Primary::parse(primary)))
            .map_prefix(|unary_op, mut expr| {
                expr.unary_op.insert_str(0, unary_op.as_str());
                expr
            })
            .map_infix(|mut lhs, op, rhs| {
                lhs.exprs.extend(rhs.with_op(Operation::parse(op)));
                lhs
            })
            .parse(pair.into_inner())
    }
}

impl core::fmt::Display for Expression {
    #[expect(
        clippy::needless_borrowed_reference,
        reason = "I personally prefer to enforce `clippy::pattern-type-mismatch`"
    )]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.unary_op, self.primary)?;
        for &(ref op, ref unary_op, ref rhs) in &self.exprs {
            write!(f, " {op} {unary_op}{rhs}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
enum Primary {
    // TODO(fh): expand to `constant` and `builtin_fn`
    Assertion(Box<Assertion>),
    Atom(String),
    Expr(Box<Expression>),
}

impl AbstractSyntaxTreeNode for Primary {
    fn parse(primary: Pair<'_, Rule>) -> Self {
        assert!(matches!(
            primary.as_rule(),
            Rule::Assertion | Rule::atom | Rule::expr
        ));

        match primary.as_rule() {
            Rule::Assertion => Self::Assertion(Box::new(Assertion::parse(primary))),
            Rule::atom => Self::Atom(String::parse(primary)),
            Rule::expr => Self::Expr(Box::new(Expression::parse(primary))),
            _ => unreachable!(),
        }
    }
}

impl core::fmt::Display for Primary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Assertion(ref assertion) => write!(f, "{assertion}"),
            Self::Atom(ref inner) => write!(f, "{inner}"),
            Self::Expr(ref expr) => write!(f, "({expr})"),
        }
    }
}

#[derive(Debug, From, Display)]
enum Operation {
    // TODO(fh): use proper struct
    Op(String),
    // Sub(Sub)
    // Add(Add)
}
