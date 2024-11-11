#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

type ColorType = u8;
pub const SCALE_FACTOR: u32 = 100_000;
pub const MAX_COLOR: usize = ColorType::MAX as _;

#[derive(Debug)]
struct ColorMap {
    data: Box<[ColorType]>,
    height: usize,
    width: usize,
}

impl ColorMap {
    fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0; width * height].into_boxed_slice(),
            height,
            width,
        }
    }
}

impl core::ops::IndexMut<usize> for ColorMap {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < self.height);
        let begin = i * self.width;
        let end = begin + self.width;
        &mut self.data[begin..end]
    }
}

impl core::ops::Index<usize> for ColorMap {
    type Output = [ColorType];

    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < self.height);
        let begin = i * self.width;
        let end = begin + self.width;
        &self.data[begin..end]
    }
}

#[derive(Debug)]
pub struct PixelGenerator {
    colors: [ColorMap; 3],
    radius: isize,
    height: usize,
    width: usize,
    probability: u32,
    rng: Pcg64Mcg,
}

impl PixelGenerator {
    #[must_use]
    pub fn new(width: usize, height: usize, radius: isize, probability: u32, seed: u64) -> Self {
        Self {
            colors: core::array::from_fn(|_| ColorMap::new(width, height)),
            radius,
            height,
            width,
            probability,
            rng: Pcg64Mcg::seed_from_u64(seed),
        }
    }

    fn color(&mut self, i: usize, j: usize, color: usize) -> ColorType {
        if self.colors[color][i][j] == 0 {
            self.colors[color][i][j] = if self.rng.gen_range(0..SCALE_FACTOR) < self.probability {
                self.rng.gen()
            } else {
                let di = self.rng.gen_range(0..=self.radius);
                let dj = self.rng.gen_range(0..=self.radius);
                // let di = self.rng.gen_range(-self.radius..=self.radius);
                // let dj = self.rng.gen_range(-self.radius..=self.radius);
                self.color(
                    i.saturating_add_signed(di) % self.height,
                    j.saturating_add_signed(dj) % self.width,
                    color,
                )
            };
        }
        self.colors[color][i][j]
    }

    pub fn red(&mut self, i: usize, j: usize) -> ColorType {
        self.color(i, j, 0)
    }

    pub fn green(&mut self, i: usize, j: usize) -> ColorType {
        self.color(i, j, 1)
    }

    pub fn blue(&mut self, i: usize, j: usize) -> ColorType {
        self.color(i, j, 2)
    }
}
