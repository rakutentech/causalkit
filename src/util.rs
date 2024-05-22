use std::fs::File;
use csv::Writer;
use std::error::Error;
use std::env;

use crate::config::Float;

pub fn write_csv(data: &Vec<Vec<Float>>, fname: &str) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(fname)?;
    let mut wtr = Writer::from_writer(&mut file);
    for row in data {
        let row: Vec<String> = row.iter().map(|x| x.to_string()).collect();
        wtr.write_record(row)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn parse_arg(pos: usize) -> String {
    let nth_arg = env::args().nth(pos);

    let val: String = match nth_arg {
        Some(arg) => arg,
        None => {panic! ("parse_arg failed")},
    };

    val
}
