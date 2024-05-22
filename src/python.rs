use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::rf::RandomForest;
use crate::config::{Float, TreeConfig};
use crate::tree::{Tree, ClassificationTree, RegressionTree};
use crate::strategy::Strategy;
use crate::kl::KLStrategy;
use crate::regression::RegressionStrategy;
use crate::dmatrix::DMatrix;
use crate::data_loader::DataLoader;

trait CausalModelInterface {
    fn get_loader(&mut self) -> &mut DataLoader;
    fn to_string(& self) -> String;
    fn save(& self, fname: &String) -> std::io::Result<()>;
    fn fit(&mut self, m: DMatrix);
    fn predict(&mut self, m: DMatrix) -> Vec<Vec<Float>>;
}

trait CausalModelFactory {
    fn new() -> Self;
    fn create<T: Tree + Clone + std::marker::Send + 'static, S: Strategy + Clone + std::marker::Send + 'static>(& self, config: &PyDict) -> Box<dyn CausalModelInterface + Send>;
    fn from_string<T: Tree + Clone + std::marker::Send + 'static, S: Strategy + Clone + std::marker::Send + 'static>(& self, ss: &String) -> Box<dyn CausalModelInterface + Send>;
    fn load<T: Tree + Clone + std::marker::Send + 'static, S: Strategy + Clone + std::marker::Send + 'static>(& self, fname: &String) -> Box<dyn CausalModelInterface + Send>;
}

pub struct RandomForestFactory {}


pub fn extract_vec(conf: &PyDict, key: &str) -> Vec<String> {
    match conf.get_item(key) {
        Ok(Some(item)) => {
            // Assuming the item is a list of strings, you need to convert it to Vec<String>
            if let Ok(strings) = item.extract::<Vec<String>>() {
                strings
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

pub fn extract_float(conf: &PyDict, key: &str, val: Float) -> Float {
    match conf.get_item(key) {
        Ok(Some(item)) => {
            // Assuming the item is a list of strings, you need to convert it to Vec<String>
            if let Ok(num) = item.extract::<f64>() {
                num as Float
            } else {
                val
            }
        }
        _ => val,
    }
}

pub fn extract_usize(conf: &PyDict, key: &str, val: usize) -> usize {
    match conf.get_item(key) {
        Ok(Some(item)) => {
            // Assuming the item is a list of strings, you need to convert it to Vec<String>
            if let Ok(num) = item.extract::<usize>() {
                num
            } else {
                val
            }
        }
        _ => val,
    }
}

pub fn extract_bool(conf: &PyDict, key: &str, val: bool) -> bool {
    match conf.get_item(key) {
        Ok(Some(item)) => {
            // Assuming the item is a list of strings, you need to convert it to Vec<String>
            if let Ok(num) = item.extract::<bool>() {
                num
            } else {
                val
            }
        }
        _ => val,
    }
}

pub fn extract_string(conf: &PyDict, key: &str, val: String) -> String {
    match conf.get_item(key) {
        Ok(Some(item)) => {
            // Assuming the item is a list of strings, you need to convert it to Vec<String>
            if let Ok(num) = item.extract::<String>() {
                num
            } else {
                val
            }
        }
        _ => val,
    }
}

impl CausalModelFactory for RandomForestFactory {

    fn new() -> Self {
        let factory = RandomForestFactory {};
        factory
    }

    fn create<T: Tree + Clone + std::marker::Send + 'static, S: Strategy + Clone + std::marker::Send + 'static>(& self, conf: &PyDict) -> Box<dyn CausalModelInterface + Send> {
        let index_cols = extract_vec(conf, "index");
        let feature_cols = extract_vec(conf, "feature");
        let cat_cols = extract_vec(conf, "cat");
        let treatment_cols = extract_vec(conf, "treatment");
        let y_col = extract_string(conf, "y", "".to_string());
        let weight_col = extract_string(conf, "weight", "".to_string());
        let n_bin = extract_usize(conf, "n_bin", 30);
        let min_samples_leaf = extract_usize(conf, "min_samples_leaf", 100);
        let min_samples_treatment = extract_usize(conf, "min_samples_treatment", 10);
        let n_reg = extract_usize(conf, "n_reg", 10);
        let alpha = extract_float(conf, "alpha", 0.9 as Float);
        let normalization = extract_bool(conf, "normalization", true);
        let max_features = extract_usize(conf, "max_features", 10);
        let max_depth = extract_usize(conf, "max_depth", 6);
        let n_tree = extract_usize(conf, "n_tree", 100);
        let subsample = extract_float(conf, "subsample", 1.0);
        let n_thread = extract_usize(conf, "n_thread", 1);

        let seed = extract_usize(conf, "seed", usize::MAX);
        let seed = match seed {
            usize::MAX => None,
            _ => Some(seed as u64)
        };

        let conf = TreeConfig {
            index_cols, feature_cols, cat_cols, treatment_cols, y_col, weight_col,
            n_bin, min_samples_leaf, min_samples_treatment, n_reg, alpha, normalization,
            max_features, max_depth, n_tree, subsample, n_thread, seed
        };

        let model: RandomForest<T, S> = RandomForest::new(conf);

        let interface: RandomForestInterface<T, S> = RandomForestInterface { model };
        Box::new(interface) as Box<dyn CausalModelInterface + Send>
    }

    fn from_string<T: Tree + Clone + std::marker::Send + 'static, S: Strategy + Clone + std::marker::Send + 'static>(& self, ss: &String) -> Box<dyn CausalModelInterface + Send> {
        let model: RandomForest<T, S> = RandomForest::from_string(ss);
        let interface: RandomForestInterface<T, S> = RandomForestInterface { model };
        Box::new(interface) as Box<dyn CausalModelInterface + Send>
    }

    fn load<T: Tree + Clone + std::marker::Send + 'static, S: Strategy + Clone + std::marker::Send + 'static>(& self, fname: &String) -> Box<dyn CausalModelInterface + Send> {
        let model: RandomForest<T, S> = RandomForest::load(fname).unwrap();
        let interface: RandomForestInterface<T, S> = RandomForestInterface { model };
        Box::new(interface) as Box<dyn CausalModelInterface + Send>
    }
}

pub struct RandomForestInterface<T: Tree + Clone + std::marker::Send, S: Strategy + Clone + std::marker::Send> {
    model: RandomForest<T, S>
}

impl<T: Tree + Clone + std::marker::Send + 'static, S: Strategy + Clone + std::marker::Send + 'static> 
    CausalModelInterface for RandomForestInterface<T, S> {

    fn get_loader(&mut self) -> &mut DataLoader {
        &mut self.model.loader
    }

    fn to_string(& self) -> String {
        self.model.to_string()
    }

    fn save(& self, fname: &String) -> std::io::Result<()> {
        self.model.save(fname)
    }

    fn fit(&mut self, m: DMatrix) {
        self.model.fit(m);
    }

    fn predict(&mut self, m: DMatrix) -> Vec<Vec<Float>> {
        self.model.predict(m)
    }
}

#[pyclass]
pub struct CausalModel {
    ptr: Box<dyn CausalModelInterface + Send>,
}

#[pymethods]
impl CausalModel {

    #[new]
    pub fn new(name: String, conf: &PyDict) -> CausalModel {
        let name_str = name.as_str();
        let ptr = match name_str {
            "RandomForestClassifier" => {
                let factory = RandomForestFactory::new();
                let interface = factory.create::<ClassificationTree, KLStrategy>(conf);
                Some(interface)
            },
            "RandomForestRegressor" => {
                let factory = RandomForestFactory::new();
                let interface = factory.create::<RegressionTree, RegressionStrategy>(conf);
                Some(interface)
            },
            &_ => {panic!("model name {} not found", name);}
        };

        let ptr = ptr.unwrap();
        CausalModel { ptr }
    }

    pub fn load(&self, name: String, path: String) -> CausalModel {
        let name_str = name.as_str();
        let ptr = match name_str {
            "RandomForestClassifier" => {
                let factory = RandomForestFactory::new();
                let interface = factory.load::<ClassificationTree, KLStrategy>(&path);
                Some(interface)
            },
            "RandomForestRegressor" => {
                let factory = RandomForestFactory::new();
                let interface = factory.load::<RegressionTree, RegressionStrategy>(&path);
                Some(interface)
            },
            &_ => {panic!("model name {} not found", name);}
        };
        let ptr = ptr.unwrap();
        CausalModel { ptr }
    }

    pub fn save(&self, path: String) -> () {
        (*self.ptr).save(&path).unwrap()
    }

    pub fn fit(& mut self, headers: Vec<String>, arr: Vec<Vec<Option<Float>>>) -> () {
        let indices: Vec<Vec<String>> = Vec::new();
        let loader = (*self.ptr).get_loader();
        let m = loader.from_memory(&headers, &indices, &arr);
        (*self.ptr).fit(m);
    }

    pub fn predict(& mut self, headers: Vec<String>, arr: Vec<Vec<Option<Float>>>) -> Vec<Vec<Float>> {
        let indices: Vec<Vec<String>> = Vec::new();
        let loader = (*self.ptr).get_loader();
        let m = loader.from_memory(&headers, &indices, &arr);
        let score = (*self.ptr).predict(m);
        score
    }
}

#[pymodule]
fn causalkit(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CausalModel>()?;
    Ok(())
}
