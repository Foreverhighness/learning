#![feature(non_null_from_ref)]

#[cfg(not(feature = "rayon"))]
use std::thread as runtime;

use behavior::basic::{CownPtr, run_when};
use crossbeam_channel::{Receiver, Sender, bounded};
use rand::{Rng, thread_rng};
#[cfg(feature = "rayon")]
use rayon as runtime;

mod behavior;

#[cfg(test)]
mod tests;

fn main() {
    const ITER: usize = 4;
    const LOGSZ_LO: usize = 10;
    const LOGSZ_HI: usize = 13;

    let (senders, receivers): (Vec<Sender<()>>, Vec<Receiver<()>>) = (0..ITER).map(|_| bounded(1)).unzip();
    let mut rng = thread_rng();

    for (i, sender) in senders.into_iter().enumerate() {
        let logsize = LOGSZ_LO + i % (LOGSZ_HI - LOGSZ_LO);
        let len = 1 << logsize;
        let mut arr = (0..len).map(|_| rng.r#gen()).collect::<Vec<_>>();
        runtime::spawn(move || {
            let res = merge_sort(arr.clone());
            arr.sort_unstable();
            assert_eq!(arr, res);
            sender.send(()).unwrap();
        });
    }

    for receiver in receivers {
        receiver.recv().unwrap();
    }
}

fn merge_sort_inner(
    idx: usize,
    step_size: usize,
    n: usize,
    boc_arr: &[CownPtr<usize>],
    boc_finish: &[CownPtr<usize>],
    sender: &Sender<Vec<usize>>,
) {
    if idx == 0 {
        return;
    }

    // Recursively sort a subarray within range [from, to).
    let from = idx * step_size - n;
    let to = (idx + 1) * step_size - n;

    let mut bocs = boc_arr[from..to].to_vec();
    bocs.push(boc_finish[idx].clone());
    bocs.push(boc_finish[idx * 2].clone());
    bocs.push(boc_finish[idx * 2 + 1].clone());

    let boc_arr: Box<[_]> = boc_arr.into();
    let boc_finish: Box<[_]> = boc_finish.into();
    let sender = sender.clone();

    run_when(bocs, move |mut content| {
        let left_and_right_sorted = (*content[step_size + 1] == 1) && (*content[step_size + 2] == 1);
        if !left_and_right_sorted || *content[step_size] == 1 {
            // If both subarrays are not ready or we already sorted for this range, skip.
            return;
        }

        // Now, merge the two subarrays.
        let mut lo = 0;
        let mut hi = step_size / 2;
        let mut res = Vec::new();
        while res.len() < step_size {
            if lo >= step_size / 2 || (hi < step_size && *content[lo] > *content[hi]) {
                res.push(*content[hi]);
                hi += 1;
            } else {
                res.push(*content[lo]);
                lo += 1;
            }
        }
        for i in 0..step_size {
            *content[i] = res[i];
        }

        // Signal that we have sorted the subarray [from, to).
        *content[step_size] = 1;

        // If the sorting process is completed send a signal to the main thread.
        if idx == 1 {
            sender.send(res).unwrap();
            return;
        }

        // Recursively sort the larger subarray (bottom up)
        merge_sort_inner(idx / 2, step_size * 2, n, &boc_arr, &boc_finish, &sender);
    });
}

/// Sorts and returns a sorted version of `array`.
///
/// # Panics
///
/// No panic
// TODO: We could make this generic over `T : Ord + Send`, but it might also need `Default` or
// usage of `MabyeUninit`.
pub fn merge_sort(array: Vec<usize>) -> Vec<usize> {
    let n = array.len();
    if n == 1 {
        return array;
    }

    let boc_arr: Box<[CownPtr<usize>]> = array.into_iter().map(CownPtr::new).collect();
    let boc_finish: Box<[CownPtr<usize>]> = (0..(2 * n)).map(|_| CownPtr::new(0)).collect();

    let (finish_sender, finish_receiver) = bounded(0);

    for i in 0..n {
        let c_finished = boc_finish[i + n].clone();

        let boc_arr_clone = boc_arr.clone();
        let boc_finish_clone = boc_finish.clone();
        let finish_sender = finish_sender.clone();
        when!(c_finished; finished; {
            // Signal that the sorting of subarray for [i, i+1) is finished.
            *finished = 1;
            merge_sort_inner((n + i) / 2, 2, n, &boc_arr_clone, &boc_finish_clone, &finish_sender);
        });
    }

    // Wait until sorting finishes and get the result.
    finish_receiver.recv().unwrap()
}
