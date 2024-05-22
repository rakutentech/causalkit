use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::error::Error;
use std::cell::RefCell;
use csv::Reader;

use crate::config::{Int, Float};
use crate::bin::{DiscreteBin, ContinuousBin};
use crate::dmatrix::DMatrix;

#[derive(Default, Clone, Debug)]
pub struct DataLoader {
    pub features: Vec<String>,
    pub response: String,
    pub treatments: Vec<String>,
    pub weight: String,
    pub bins_cont: HashMap<String, ContinuousBin>,
    pub bins_disc: HashMap<String, DiscreteBin>,
    pub cats: Vec<String>,
    pub n_bin: usize,
}

impl DataLoader {

    pub fn new(features: Vec<String>, response: String, treatments: Vec<String>, 
        weight: String, cats: Vec<String>, n_bin: usize, bins_cont: HashMap<String, ContinuousBin>,
        bins_disc: HashMap<String, DiscreteBin>) -> DataLoader {
        if n_bin >= (Int::MAX as usize) { 
            panic! ("n_bin {} exceeds max {}", n_bin, Int::MAX);
        }
        DataLoader {features, response, treatments, weight, bins_cont, bins_disc, cats, n_bin}
    }

    pub fn from_memory(&mut self, headers: &Vec<String>, indices: &Vec<Vec<String>>, 
        arr: &Vec<Vec<Option<Float>>>) -> DMatrix {

        let feature_pos = DataLoader::find_pos(headers, &self.features, true);
        let response_pos = DataLoader::find_pos_single(headers, &self.response, false);
        let treatment_pos = DataLoader::find_pos(headers, &self.treatments, false);
        let weight_pos = DataLoader::find_pos_single(headers, &self.weight, false);

        let mut is_bool = Vec::new();
        for col in self.features.iter() {
            let isin = self.cats.contains(col);
            is_bool.push(isin);
        }

        let mut feature: Vec<Vec<Option<Int>>> = Vec::new();
        for (idx, pos) in feature_pos.iter().enumerate() {
            let p = pos.unwrap();
            let v = DataLoader::get_nth_column(arr, p);
            let name = self.features.get(idx).unwrap().clone();

            let iscat = is_bool.get(idx).unwrap();
            let vt = match iscat {
                x if *x => {
                    if !self.bins_disc.contains_key(& name) {
                        let mut bin = DiscreteBin::new(self.n_bin);
                        bin.fit(&v);
                        self.bins_disc.insert(name.clone(), bin);
                    }
    
                    self.bins_disc.get(& name).unwrap().map(&v)
                },
                _ => {
                    if !self.bins_cont.contains_key(& name) {
                        let mut bin = ContinuousBin::new(self.n_bin, None);
                        bin.fit(&v);
                        self.bins_cont.insert(name.clone(), bin);
                    }
    
                    self.bins_cont.get(& name).unwrap().map(&v)
                }
            };

            feature.push(vt);
        }

        let mut response: Vec<Float> = Vec::new();
        if !response_pos.is_none() {
            let v = DataLoader::get_nth_column(arr, response_pos.unwrap());
            response = v.into_iter().map(|option| option.unwrap()).collect();
        }

        let mut treatments: Vec<Vec<Int>> = Vec::new();
        let mut treatment_size: Vec<usize> = Vec::new();
        if treatment_pos.len() > 0 && !treatment_pos.get(0).unwrap().is_none() {
            for pos in treatment_pos.iter() {
                let v = DataLoader::get_nth_column(arr, pos.unwrap());
                let v: Vec<Int> = v.into_iter().map(|option| option.unwrap() as Int).collect();
                let size = v.iter().max().unwrap() + 1;
                treatments.push(v);
                treatment_size.push(size as usize);
            }
        }

        let size = arr.len();
        let weights: Vec<Float> = match weight_pos {
            Some(pos) => {
                let v = DataLoader::get_nth_column(arr, pos);
                v.into_iter().map(|option| option.unwrap()).collect()
            },
            None => { vec![1.0; size] }
        };

        let name = self.features.clone();
        let bin_size = self.get_bin_size();
        DMatrix::new(indices.clone(), feature, response, treatments, weights, is_bool, name, 
            bin_size, treatment_size)
    }

    pub fn from_csv(&mut self, path: String) -> DMatrix {
        let rdr = DataLoader::freader(path.as_str()).unwrap();
        let headers = DataLoader::get_header(&mut rdr.borrow_mut()).unwrap();
        let n_col = headers.len();
        let arr = DataLoader::get_content(&mut rdr.borrow_mut(), n_col).unwrap();
        let indices = Vec::new();

        self.from_memory(&headers, &indices, &arr)
    }

    pub fn get_bin_size(& self) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for (name, bin) in self.bins_cont.iter() {
            map.insert(name.clone(), bin.n_bin);
        }

        for (name, bin) in self.bins_disc.iter() {
            map.insert(name.clone(), bin.n_bin);
        }

        map
    }

    fn find_pos(headers: &Vec<String>, fnames: &Vec<String>, required: bool) -> Vec<Option<usize>> {
        let mut index: Vec<Option<usize>> = Vec::new();
        for fname in fnames.iter() {
            let pos = headers.iter().position(|name| name == fname);
            if pos.is_none() && required {panic!("Column is missing: {}", fname)}
            index.push(pos);
        }
        index
    }

    fn find_pos_single(headers: &Vec<String>, fname: &String, required: bool) -> Option<usize> {
        let pos = headers.iter().position(|name| name == fname);
        if pos.is_none() && required {panic!("Column is missing: {}", fname)}
        pos
    }

    fn get_nth_column(arr: &Vec<Vec<Option<Float>>>, n: usize) -> Vec<Option<Float>> {
        arr.iter().map(|x| *x.get(n).unwrap()).collect::<Vec<_>>()
    }

    fn freader(fpath: &str) -> Result<RefCell<Reader<File>>, Box<dyn Error>> {
        let path = Path::new(fpath);
        let rdr = Reader::from_path(path)?;
        Ok(RefCell::new(rdr))
    }

    fn get_header(rdr: &mut Reader<File>) -> Result<Vec<String>, Box<dyn Error>> {
        let headers = rdr.headers()?;
        let mut h = Vec::new();

        for header in headers.iter() {
            h.push(header.to_string());
        }
        Ok(h)
    }

    fn get_content(rdr: &mut Reader<File>, n_col: usize) -> Result<Vec<Vec<Option<Float>>>, 
        Box<dyn Error>> {

        let mut data: Vec<Vec<Option<Float>>> = Vec::new();

        for result in rdr.records() {
            let record = result?;
            let l = record.len();
            if l != n_col {
                panic! ("invalid row encountered")
            }

            let row: Vec<Option<Float>> = record.iter().map(|field| {
                let empty = field.is_empty();
                match empty {
                    x if x => None,
                    _ => match field.parse::<Float>() {
                        Ok(num) => Some(num as Float),
                        Err(_) => None
                    }
                }
            }).collect();

            data.push(row);
        }

        Ok(data)
    }
}
