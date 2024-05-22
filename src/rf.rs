use std::fs::File;
use rand::{rngs::StdRng, SeedableRng};
use std::sync::{Arc, Mutex};
use std::vec;
use std::thread;
use std::io::prelude::*;
use std::collections::HashMap;

use crate::data_loader::DataLoader;
use crate::bin::{DiscreteBin, ContinuousBin};
use crate::tree::Tree;
use crate::strategy::Strategy;
use crate::config::{Float, TreeConfig};
use crate::learn::TreeLearn;
use crate::dmatrix::DMatrix;
use crate::random::Random;
use crate::linalg::Matrix;


pub struct RandomForest<T: Tree + Clone + std::marker::Send, S: Strategy + Clone + std::marker::Send> {
    pub conf: TreeConfig,
    pub loader: DataLoader, 
    pub learners: Vec<TreeLearn<T, S>>
}

impl<T: Tree + Clone + std::marker::Send + 'static, S: Strategy + Clone + std::marker::Send + 'static> RandomForest<T, S> {

    pub fn new(conf: TreeConfig) -> RandomForest<T, S> {
        let bins_cont: HashMap<String, ContinuousBin> = HashMap::new();
        let bins_disc: HashMap<String, DiscreteBin> = HashMap::new();

        let loader = DataLoader::new(conf.feature_cols.clone(), conf.y_col.clone(), conf.treatment_cols.clone(),
            conf.weight_col.clone(), conf.cat_cols.clone(), conf.n_bin, bins_cont, bins_disc);

        let learners: Vec<TreeLearn<T, S>> = Vec::new();
        let model: RandomForest<T, S> = RandomForest {
            conf,
            loader,
            learners
        };

        model
    }

    pub fn from_string(ss: &String) -> RandomForest<T, S> {
        let lines: Vec<&str> = ss.split("\n\n").collect();

        let line = lines.get(0).unwrap();
        let conf: TreeConfig = serde_json::from_str(& line.to_string()).unwrap();

        let mut bins_cont: HashMap<String, ContinuousBin> = HashMap::new();
        let mut bins_disc: HashMap<String, DiscreteBin> = HashMap::new();

        let n_feature = conf.feature_cols.len();
        for n in 0..n_feature {
            let line = lines.get(n + 1).unwrap();
            let fields: Vec<&str> = line.split('\n').collect();
            let bin_type = fields.get(0).unwrap();
            let name = fields.get(1).unwrap();
            let content = fields.get(2).unwrap();

            if bin_type.to_string() == String::from("Continuous") {
                let bin: ContinuousBin = serde_json::from_str(& content.to_string()).unwrap();
                bins_cont.insert(name.to_string(), bin);
            } else if bin_type.to_string() == String::from("Discrete") {
                let bin: DiscreteBin = serde_json::from_str(& content.to_string()).unwrap();
                bins_disc.insert(name.to_string(), bin);
            } else {
                panic! ("bin_type {} unknown", bin_type);
            }
        }

        let loader = DataLoader::new(conf.feature_cols.clone(), conf.y_col.clone(), conf.treatment_cols.clone(),
            conf.weight_col.clone(), conf.cat_cols.clone(), conf.n_bin, bins_cont, bins_disc);

        let mut learners: Vec<TreeLearn<T, S>> = Vec::new();
        let n_tree = conf.n_tree;
        for n in 0..n_tree {
            let line = lines.get(n + 1 + n_feature).unwrap();
            let learner: TreeLearn<T, S> = TreeLearn::from_string(& line.to_string());
            learners.push(learner);
        }

        let model: RandomForest<T, S> = RandomForest {
            conf,
            loader,
            learners
        };

        model
    }

    pub fn to_string(& self) -> String {
        let mut ss = String::new();

        let line = serde_json::to_string(& self.conf).unwrap();
        let line = format!("{}\n\n", line);
        ss = ss + &line;

        for (name, bin) in self.loader.bins_cont.iter() {
            let line = serde_json::to_string(& bin).unwrap();
            let line = format!("Continuous\n{}\n{}\n\n", name, line);
            ss = ss + &line;
        }
        for (name, bin) in self.loader.bins_disc.iter() {
            let line = serde_json::to_string(& bin).unwrap();
            let line = format!("Discrete\n{}\n{}\n\n", name, line);
            ss = ss + &line;
        }

        for learner in self.learners.iter() {
            let line = learner.to_string();
            let line = format!("{}\n\n", line);
            ss = ss + &line;
        }

        ss
    }

    pub fn load(fname: &String) -> std::io::Result<RandomForest<T, S>> {
        let mut file = File::open(fname.as_str())?;
        let mut ss = String::new();
        file.read_to_string(&mut ss)?;

        let m: RandomForest<T, S> = RandomForest::from_string(&ss);

        Ok(m)
    }

    pub fn save(& self, fname: &String) -> std::io::Result<()> {
        let ss = self.to_string();
        let mut file = File::create(fname)?;
        file.write_all(ss.as_bytes())?;

        Ok(())
    }

    pub fn fit(&mut self, m: DMatrix) {
        let learners = match self.conf.n_thread {
            n if n == 1 => self.fit_seq(m),
            _ => self.fit_par(m)
        };

        self.learners.extend(learners);
    }

    pub fn predict(&mut self, m: DMatrix) -> Vec<Vec<Float>> {
        match self.conf.n_thread {
            n if n == 1 => self.predict_seq(m),
            _ => self.predict_par(m)
        }
    }

    fn fit_seq(&mut self, m: DMatrix) -> Vec<TreeLearn<T, S>> {
        let seed = self.conf.seed.clone();
        let rng: StdRng = match seed {
            None => StdRng::from_entropy(),
            Some(v) => SeedableRng::seed_from_u64(v)
        };

        let rng_share = Arc::new(Mutex::new(rng));
        let mut random = Random::new(rng_share);

        let mut learners = Vec::new();

        for _ in 0..self.conf.n_tree {
            let mut learn: TreeLearn<T, S> = TreeLearn::new(& self.conf);
            learn.fit(&m, & self.conf, &mut random);
            learners.push(learn);
        }

        learners
    }

    fn fit_par_thread(k: usize, conf: Arc<TreeConfig>, m: Arc<DMatrix>, 
        random: &mut Random) -> Vec<TreeLearn<T, S>> {

        let mut learners = Vec::new();

        for _ in 0..k {
            let mut learn: TreeLearn<T, S> = TreeLearn::new(&(*conf));
            learn.fit(&(*m), &(*conf), random);
            learners.push(learn);
        }

        learners
    }

    fn fit_par(&mut self, m: DMatrix) -> Vec<TreeLearn<T, S>> {
        let n_tree = self.conf.n_tree;
        let n_thread = self.conf.n_thread;

        let mut jobs: Vec<usize> = Vec::new();

        if n_tree < n_thread {
            jobs.extend(vec![1; n_tree]);
        } else {
            let q: usize = n_tree / n_thread;
            let r: usize = n_tree % n_thread;
    
            let mut jobs_ = vec![q; n_thread];
            for i in 0..r {
                jobs_[i] += 1;
            }
            jobs.extend(jobs_)
        }

        let mut learners = Vec::new();

        let seed = self.conf.seed.clone();
        let rng: StdRng = match seed {
            None => StdRng::from_entropy(),
            Some(v) => SeedableRng::seed_from_u64(v)
        };

        let m_share = Arc::new(m);
        let conf = self.conf.clone();
        let conf_share = Arc::new(conf);
        let rng_share = Arc::new(Mutex::new(rng));

        let mut handles = Vec::new();
        for k in jobs.iter() {
            let k_c = k.clone();
            let d = Arc::clone(&m_share);
            let c = Arc::clone(&conf_share);
            let r_c = Arc::clone(&rng_share);
            let mut r = Random::new(r_c);

            let handle = thread::spawn(move || {
                RandomForest::fit_par_thread(k_c, c, d, &mut r)
            });

            handles.push(handle);
        }

        for h in handles {
            let result = h.join().unwrap();
            learners.extend(result);
        }

        learners
    }

    pub fn predict_seq(&mut self, m: DMatrix) -> Vec<Vec<Float>> {
        let mut avg: Option<Matrix<Float>> = None;

        for learn in self.learners.iter_mut() {
            let score = learn.predict(&m);
            let arr: Matrix<Float> = Matrix::new(& score);

            if avg.is_none() {
                avg = Some(arr);
                continue;
            }

            let arr = arr.add(& avg.unwrap());
            avg = Some(arr);
        }

        let avg = avg.unwrap();
        let avg = avg.divide_scalar(self.learners.len() as Float);
        avg.get_data().clone()
    }

    fn split_job_predict(& self, n_tree: usize, n_thread: usize) -> Vec<Vec<usize>> {
        if n_tree < n_thread {
            let v: Vec<usize> = (0..n_tree).collect();
            let mut vs = Vec::new();
            for i in v {
                vs.push(vec![i]);
            }

            return vs;
        }

        let q: usize = n_tree / n_thread;
        let r: usize = n_tree % n_thread;

        let mut vs = Vec::new();
        for i in 0..n_thread {
            let start = match i {
                i if i < r => i * (q + 1),
                i => r * (q + 1) + (i - r) * q
            };

            let size = match i {
                i if i < r => q + 1,
                _ => q
            };

            let end = start + size;
            let v: Vec<usize> = (start..end).collect();
            vs.push(v);
        }

        vs
    }

    fn predict_par_thread(learners: &mut Vec<TreeLearn<T, S>>, m: Arc<DMatrix>) -> Vec<Vec<Float>> {
        let mut avg: Option<Matrix<Float>> = None;

        for learn in learners.iter_mut() {
            let score = (*learn).predict(&m);
            let arr: Matrix<Float> = Matrix::new(& score);

            if avg.is_none() {
                avg = Some(arr);
                continue;
            }

            let arr = arr.add(& avg.unwrap());
            avg = Some(arr);
        }

        avg.unwrap().get_data().clone()
    }

    pub fn predict_par(&mut self, m: DMatrix) -> Vec<Vec<Float>> {
        let jobs: Vec<Vec<usize>> = self.split_job_predict(self.conf.n_tree, self.conf.n_thread);

        let m_share = Arc::new(m);

        let mut handles = Vec::new();
        for k in jobs {
            let mut learners: Vec<TreeLearn<T, S>> = Vec::new();
            for i in k.iter() {
                let learner: TreeLearn<T, S> = self.learners.get(*i).unwrap().clone();
                learners.push(learner);
            }

            let d = Arc::clone(&m_share);

            let handle = thread::spawn(move || {
                RandomForest::predict_par_thread(&mut learners, d)
            });

            handles.push(handle);
        }

        let mut avg: Option<Matrix<Float>> = None;

        for h in handles {
            let score = h.join().unwrap();
            let arr: Matrix<Float> = Matrix::new(& score);

            if avg.is_none() {
                avg = Some(arr);
                continue;
            }

            let arr = arr.add(& avg.unwrap());
            avg = Some(arr);
        }

        let avg = avg.unwrap();
        let avg = avg.divide_scalar(self.learners.len() as Float);
        avg.get_data().clone()
    }
}
