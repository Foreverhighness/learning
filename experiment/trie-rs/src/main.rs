mod trie;
mod trie_node;

use trie::Trie;

fn main() {
    {
        let mut trie = Trie::new();

        trie = trie.put::<i32>("test-int", 233);
        dbg!(&trie);
        trie = trie.put::<i32>("test-int2", 2_333_333);
        dbg!(&trie);
        trie = trie.put::<String>("test-string", String::from("test"));
        dbg!(&trie);
        trie = trie.put::<&str>("", "empty-key");
        dbg!(&trie);

        debug_assert_eq!(trie.get::<i32>("test-int"), Some(&233));
        debug_assert_eq!(trie.get::<i32>("test-int2"), Some(&2_333_333));
        debug_assert_eq!(
            trie.get::<String>("test-string"),
            Some(&String::from("test"))
        );
        debug_assert_eq!(trie.get::<&str>(""), Some(&"empty-key"));
    }

    {
        let mut trie = Trie::new();
        trie = trie.put::<u32>("test", 233);
        debug_assert_eq!(trie.get::<u32>("test"), Some(&233));
        // Put something else
        trie = trie.put::<u32>("test", 23333333);
        debug_assert_eq!(trie.get::<u32>("test"), Some(&23333333));
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
            let key = format!("{:#05}", i);
            let value = format!("value-{:#08}", i);
            trie = trie.put::<String>(&key, value);
        }
        let trie_full = trie.clone();
        for i in (0..23333).step_by(2) {
            let key = format!("{:#05}", i);
            let value = format!("new-value-{:#08}", i);
            trie = trie.put::<String>(&key, value);
        }
        let trie_override = trie.clone();
        for i in (0..23333).step_by(3) {
            let key = format!("{:#05}", i);
            trie = trie.remove(&key);
        }
        let trie_final = trie.clone();

        // verify trie_full
        for i in 0..23333 {
            let key = format!("{:#05}", i);
            let value = format!("value-{:#08}", i);
            debug_assert_eq!(trie_full.get::<String>(&key), Some(&value));
        }

        // verify trie_override
        for i in 0..23333 {
            let key = format!("{:#05}", i);
            if i % 2 == 0 {
                let value = format!("new-value-{:#08}", i);
                debug_assert_eq!(trie_override.get::<String>(&key), Some(&value));
            } else {
                let value = format!("value-{:#08}", i);
                debug_assert_eq!(trie_override.get::<String>(&key), Some(&value));
            }
        }

        // verify final trie
        for i in 0..23333 {
            let key = format!("{:#05}", i);
            if i % 3 == 0 {
                debug_assert_eq!(trie_final.get::<String>(&key), None);
            } else if i % 2 == 0 {
                let value = format!("new-value-{:#08}", i);
                debug_assert_eq!(trie_final.get::<String>(&key), Some(&value));
            } else {
                let value = format!("value-{:#08}", i);
                debug_assert_eq!(trie_final.get::<String>(&key), Some(&value));
            }
        }
    }
}
