use std::sync::{Arc, Mutex};
use rand::distributions::{Uniform, Distribution};


pub struct Random {
    rng: Arc<Mutex<rand::rngs::StdRng>>
}

impl Random {

    pub fn new(rng: Arc<Mutex<rand::rngs::StdRng>>) -> Random {
        Random {
            rng
        }
    }

    pub fn choose(& mut self, n: usize, k: usize, replace: bool) -> Vec<usize> {
        // if replace = True, the indices could be chosen multiple times.
        assert!(n >= k);

        let dist = Uniform::new_inclusive(0, n-1);

        let mut values = Vec::new();

        let mut g = self.rng.lock().unwrap();
        let g = &mut (*g);

        while values.len() < k {
            let sample = dist.sample(g) as usize;
            if replace || !values.contains(&sample) {
                values.push(sample);
            }
        }

        values
    }

}
