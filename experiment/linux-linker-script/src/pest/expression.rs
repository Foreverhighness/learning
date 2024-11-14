#![expect(dead_code, reason = "TODO(fh): replace `String` to `Expression`")]

#[derive(Debug)]
pub enum Expression {
    Atom(Atom),
    WithOperation(Atom, Operation, Atom),
}

#[derive(Debug)]
pub enum Atom {
    // TODO(fh): use proper struct
    Constant(String),
    BuiltinFn(String),
    Symbol(String),
    // Constant(Constant),
    // BuiltinFn(BuiltinFn),
    // Symbol(Symbol),
    Expression(Box<Expression>),
}

#[derive(Debug)]
pub enum Operation {
    // TODO(fh): use proper struct
    Op(String),
    // Sub(Sub)
    // Add(Add)
}
