mod trie;
mod trie_node;
mod trie_store;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use trie::Trie;
use trie_store::TrieStore;

#[expect(clippy::cognitive_complexity, clippy::too_many_lines, reason = "test")]
fn main() {
    {
        let mut trie = Trie::new();

        trie = trie.put::<i32>("test-int", 233);
        println!("{trie:?}");
        trie = trie.put::<i32>("test-int2", 2_333_333);
        println!("{trie:?}");
        trie = trie.put::<String>("test-string", String::from("test"));
        println!("{trie:?}");
        trie = trie.put::<&str>("", "empty-key");
        println!("{trie:?}");

        debug_assert_eq!(trie.get::<i32>("test-int"), Some(&233));
        debug_assert_eq!(trie.get::<i32>("test-int2"), Some(&2_333_333));
        debug_assert_eq!(trie.get::<String>("test-string"), Some(&String::from("test")));
        debug_assert_eq!(trie.get::<&str>(""), Some(&"empty-key"));
    }

    {
        let mut trie = Trie::new();

        trie = trie.put::<i32>("中文整数", 233);
        println!("{trie:?}");
        trie = trie.put::<i32>("中文整数二", 2_333_333);
        println!("{trie:?}");
        trie = trie.put::<String>("测试", String::from("test"));
        println!("{trie:?}");
        trie = trie.put::<&str>("😈", "emoji");
        println!("{trie:?}");

        debug_assert_eq!(trie.get::<i32>("中文整数"), Some(&233));
        debug_assert_eq!(trie.get::<i32>("中文整数二"), Some(&2_333_333));
        debug_assert_eq!(trie.get::<String>("测试"), Some(&String::from("test")));
        debug_assert_eq!(trie.get::<&str>("😈"), Some(&"emoji"));
    }

    {
        let mut trie = Trie::new();
        trie = trie.put::<u32>("test", 233);
        debug_assert_eq!(trie.get::<u32>("test"), Some(&233));
        // Put something else
        trie = trie.put::<u32>("test", 23_333_333);
        debug_assert_eq!(trie.get::<u32>("test"), Some(&23_333_333));
        // Overwrite with another type
        trie = trie.put::<&str>("test", "23333333");
        debug_assert_eq!(trie.get::<&str>("test"), Some(&"23333333"));
        // Get something that doesn't exist
        debug_assert_eq!(trie.get::<&str>("test-2333"), None);
        // Put something at root
        trie = trie.put::<&str>("", "empty-key");
        debug_assert_eq!(trie.get::<&str>(""), Some(&"empty-key"));
    }

    {
        let empty_trie = Trie::new();
        // Put something
        let trie1 = empty_trie.put::<u32>("test", 2333);
        let trie2 = trie1.put::<u32>("te", 23);
        let trie3 = trie2.put::<u32>("tes", 233);

        // Delete something
        let trie4 = trie3.remove("te");
        let trie5 = trie3.remove("tes");
        let trie6 = trie3.remove("test");

        // Check each snapshot
        debug_assert_eq!(trie3.get::<u32>("te"), Some(&23));
        debug_assert_eq!(trie3.get::<u32>("tes"), Some(&233));
        debug_assert_eq!(trie3.get::<u32>("test"), Some(&2333));

        debug_assert_eq!(trie4.get::<u32>("te"), None);
        debug_assert_eq!(trie4.get::<u32>("tes"), Some(&233));
        debug_assert_eq!(trie4.get::<u32>("test"), Some(&2333));

        debug_assert_eq!(trie5.get::<u32>("te"), Some(&23));
        debug_assert_eq!(trie5.get::<u32>("tes"), None);
        debug_assert_eq!(trie5.get::<u32>("test"), Some(&2333));

        debug_assert_eq!(trie6.get::<u32>("te"), Some(&23));
        debug_assert_eq!(trie6.get::<u32>("tes"), Some(&233));
        debug_assert_eq!(trie6.get::<u32>("test"), None);
    }

    {
        let mut trie = Trie::new();
        for i in 0..23333 {
            let key = format!("{i:#05}");
            let value = format!("value-{i:#08}");
            trie = trie.put::<String>(&key, value);
        }
        let trie_full = trie.clone();
        for i in (0..23333).step_by(2) {
            let key = format!("{i:#05}");
            let value = format!("new-value-{i:#08}");
            trie = trie.put::<String>(&key, value);
        }
        let trie_override = trie.clone();
        for i in (0..23333).step_by(3) {
            let key = format!("{i:#05}");
            trie = trie.remove(&key);
        }
        let trie_final = trie.clone();

        // verify trie_full
        for i in 0..23333 {
            let key = format!("{i:#05}");
            let value = format!("value-{i:#08}");
            debug_assert_eq!(trie_full.get::<String>(&key), Some(&value));
        }

        // verify trie_override
        for i in 0..23333 {
            let key = format!("{i:#05}");
            if i % 2 == 0 {
                let value = format!("new-value-{i:#08}");
                debug_assert_eq!(trie_override.get::<String>(&key), Some(&value));
            } else {
                let value = format!("value-{i:#08}");
                debug_assert_eq!(trie_override.get::<String>(&key), Some(&value));
            }
        }

        // verify final trie
        for i in 0..23333 {
            let key = format!("{i:#05}");
            if i % 3 == 0 {
                debug_assert_eq!(trie_final.get::<String>(&key), None);
            } else if i % 2 == 0 {
                let value = format!("new-value-{i:#08}");
                debug_assert_eq!(trie_final.get::<String>(&key), Some(&value));
            } else {
                let value = format!("value-{i:#08}");
                debug_assert_eq!(trie_final.get::<String>(&key), Some(&value));
            }
        }
    }

    {
        let store = &TrieStore::new();
        let keys_per_thread = 10_000_u32;

        std::thread::scope(move |s| {
            let mut threads = Vec::new();

            for tid in 0..4 {
                let handle = s.spawn(move || {
                    for i in 0..keys_per_thread {
                        let key = format!("{:#05}", i * 4 + tid);
                        let value = format!("value-{:#08}", i * 4 + tid);
                        store.put::<String>(&key, value);
                    }
                    for i in 0..keys_per_thread {
                        let key = format!("{:#05}", i * 4 + tid);
                        store.remove(&key);
                    }
                    for i in 0..keys_per_thread {
                        let key = format!("{:#05}", i * 4 + tid);
                        let value = format!("new-value-{:#08}", i * 4 + tid);
                        store.put::<String>(&key, value);
                    }
                });
                threads.push(handle);
            }

            let mut read_threads = Vec::new();
            let stop = Arc::new(AtomicBool::new(false));

            for tid in 0..4 {
                let stop = Arc::clone(&stop);
                let handle = s.spawn(move || {
                    let mut i = 0;
                    while !stop.load(Ordering::SeqCst) {
                        let key = format!("{:#05}", i * 4 + tid);
                        store.get::<String>(&key);
                        i = (i + 1) % keys_per_thread;
                    }
                });
                read_threads.push(handle);
            }

            for t in threads {
                let _ = t.join();
            }

            stop.store(true, Ordering::SeqCst);

            for t in read_threads {
                let _ = t.join();
            }
        });

        // verify final trie
        for i in 0..(4 * keys_per_thread) {
            let key = format!("{i:#05}");
            let value = format!("new-value-{i:#08}");
            let guard = store.get::<String>(&key).unwrap();
            debug_assert_eq!(*guard, value);
        }
    }
}
