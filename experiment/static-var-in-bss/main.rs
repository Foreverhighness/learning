#![feature(const_maybe_uninit_zeroed)]
use std::mem::MaybeUninit;

static mut S_VAR_1: u32 = 1;
static mut S_VAR_2: MaybeUninit<u32> = MaybeUninit::uninit();
static mut S_VAR_3: MaybeUninit<u32> = MaybeUninit::zeroed();

const C_VAR_1: u32 = 3;

fn foo() {
    static mut S_VAR_4: u32 = 4;
    static mut S_VAR_5: MaybeUninit<u32> = MaybeUninit::uninit();
    const C_VAR_2: u32 = 6;
    unsafe {
        println!("{:?}", S_VAR_1);
        println!("{:?}", S_VAR_2);
        println!("{:?}", S_VAR_3);
        println!("{:?}", S_VAR_4);
        println!("{:?}", S_VAR_5);
        println!("{:?}", C_VAR_1);
        println!("{:?}", C_VAR_2);

        S_VAR_1 = 11;
        println!("{:?}", S_VAR_2.assume_init());
        println!("{:?}", S_VAR_3.assume_init());

        S_VAR_4 = 44;
        S_VAR_5 = MaybeUninit::uninit();

        println!("{:?}", S_VAR_1);
        println!("{:?}", S_VAR_2);
        println!("{:?}", S_VAR_3);
        println!("{:?}", S_VAR_4);
        println!("{:?}", S_VAR_5);
        println!("{:?}", C_VAR_1);
        println!("{:?}", C_VAR_2);
    }
}

fn main() {
    foo();
}

