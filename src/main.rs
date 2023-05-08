use apache_avro::{from_value, Reader};
use arrow2::{
    array::Array,
    chunk::Chunk,
    datatypes::*,
    io::parquet::write::{
        transverse, CompressionOptions, Encoding, FileWriter, RowGroupIterator, Version,
        WriteOptions,
    },
};
use arrow2_convert::{
    serialize::{FlattenChunk, TryIntoArrow},
    ArrowDeserialize, ArrowField, ArrowSerialize,
};
use clap::Parser;
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use lazy_regex::regex_captures;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the AIDA ConLL-YAGO dataset in the TSV format.
    #[arg(long)]
    input_conll: String,
    /// Path to the mappings between Wikipedia's titles and Wikidata's QIDs in the Apache Avro format.
    #[arg(long)]
    input_wiki2qid: String,
    /// Path to the output directory.
    #[arg(long)]
    output_dir: String,
}

#[derive(Debug, Deserialize)]
struct MappingRecord {
    title: String,
    #[allow(unused)]
    pageid: u32,
    qid: Option<u32>,
}

#[derive(Debug, PartialEq, Clone)]
enum EntityType {
    OutOfDistribution,
    InDistribution(String),
    None,
}

#[derive(Debug)]
struct TokenRecord {
    document_id: u32,
    token: String,
    entity: EntityType,
}

enum Split {
    Train,
    Validation,
    Test,
}

#[derive(Debug, ArrowField, ArrowSerialize, ArrowDeserialize)]
struct Entity {
    start: u32,
    end: u32,
    pageid: Option<u32>,
    qid: Option<u32>,
}

#[derive(Debug, ArrowField, ArrowSerialize, ArrowDeserialize)]
struct DataPoint {
    uuid: String,
    document_id: u32,
    text: String,
    entities: Vec<Entity>,
}

fn parse_conll(
    path: &str,
) -> (
    (Vec<TokenRecord>, Vec<TokenRecord>, Vec<TokenRecord>),
    HashSet<String>,
) {
    let mut train = vec![];
    let mut validation = vec![];
    let mut test = vec![];

    let mut document_id = 0;
    let mut document_split = Split::Train;

    let mut titles = HashSet::new();

    let reader = BufReader::new(File::open(path).unwrap());

    for line in reader.lines() {
        let line = line.unwrap();

        if line.len() == 0 {
            continue;
        }

        let fields = line.split("\t").collect::<Vec<_>>();

        if fields.len() == 1 {
            if let Some((_, id, split)) = regex_captures!(
                r#"-DOCSTART- \(([\d]+)(testa|testb)? [^\)\\]*(?:\\.[^\)\\]*)*\)"#,
                &fields[0]
            ) {
                document_id = id.parse::<u32>().unwrap();
                document_split = match split {
                    "testa" => Split::Validation,
                    "testb" => Split::Test,
                    _ => Split::Train,
                };

                continue;
            }
        }

        let token = fields[0].nfc().collect::<String>();

        let split = match document_split {
            Split::Train => &mut train,
            Split::Validation => &mut validation,
            Split::Test => &mut test,
        };

        if fields.len() == 4 {
            split.push(TokenRecord {
                document_id,
                token,
                entity: EntityType::OutOfDistribution,
            });
        } else if fields.len() > 4 {
            let title = fields[4].chars().skip(29).nfc().collect::<String>();
            split.push(TokenRecord {
                document_id,
                token,
                entity: EntityType::InDistribution(title.clone()),
            });

            titles.insert(title);
        } else {
            split.push(TokenRecord {
                document_id,
                token,
                entity: EntityType::None,
            });
        }
    }

    ((train, validation, test), titles)
}

fn generate_dataset(
    split: Vec<TokenRecord>,
    mapping: &HashMap<String, (u32, Option<u32>)>,
) -> Vec<DataPoint> {
    let mut examples = vec![];

    for (document_id, group) in &split
        .into_iter()
        .group_by(|x| x.document_id)
    {
        let mut text = String::new();
        let mut entities = vec![];

        for (mention, group) in &group
            .map(|x| (x.token, x.entity))
            .group_by(|x| (x.clone().1))
        {
            let tokens = group.map(|x| x.0).collect::<Vec<_>>().join(" ");

            let start = (text.chars().count() + if text.is_empty() { 0 } else { 1 }) as u32;
            let end = (text.chars().count()
                + if text.is_empty() { 0 } else { 1 }
                + tokens.chars().count()) as u32;

            let mention = match mention {
                EntityType::OutOfDistribution => Some(Entity {
                    start,
                    end,
                    pageid: None,
                    qid: None,
                }),
                EntityType::InDistribution(title) => {
                    let (pageid, qid) = *mapping.get(&title).unwrap();
                    Some(Entity {
                        start,
                        end,
                        pageid: Some(pageid),
                        qid,
                    })
                }
                EntityType::None => None,
            };

            if let Some(mention) = mention {
                entities.push(mention);
            }

            if !text.is_empty() {
                text.push(' ')
            }

            text.push_str(&tokens);
        }

        examples.push(DataPoint {
            uuid: Uuid::new_v4().to_string(),
            document_id,
            text,
            entities,
        });
    }

    examples
}

fn write_dataset(split: Vec<DataPoint>, path: &str) {
    let array: Box<dyn Array> = split.try_into_arrow().unwrap();
    let array = array
        .as_any()
        .downcast_ref::<arrow2::array::StructArray>()
        .unwrap();

    let chunk = Chunk::new(vec![array.clone().boxed()]).flatten().unwrap();

    let options = WriteOptions {
        write_statistics: true,
        compression: CompressionOptions::Zstd(None),
        version: Version::V2,
        data_pagesize_limit: None,
    };

    let iter = vec![Ok(chunk)];

    let schema = Schema::from(vec![
        Field::new("uuid", DataType::Utf8, false),
        Field::new("document_id", DataType::UInt32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "entities",
            DataType::List(Box::new(Field::new(
                "",
                DataType::Struct(vec![
                    Field::new("start", DataType::UInt32, false),
                    Field::new("end", DataType::UInt32, false),
                    Field::new("pageid", DataType::UInt32, true),
                    Field::new("qid", DataType::UInt32, true),
                ]),
                false,
            ))),
            false,
        ),
    ]);

    let encodings = schema
        .fields
        .iter()
        .map(|f| transverse(&f.data_type, |_| Encoding::Plain))
        .collect();

    let row_groups =
        RowGroupIterator::try_new(iter.into_iter(), &schema, options, encodings).unwrap();

    let file = File::create(path).unwrap();

    let mut writer = FileWriter::try_new(file, schema, options).unwrap();

    for group in row_groups {
        writer.write(group.unwrap()).unwrap();
    }
    writer.end(None).unwrap();
}

fn main() {
    let args = Args::parse();

    let ((train, validation, test), titles) = parse_conll(&args.input_conll);

    let mut mapping = HashMap::new();
    // Corrections.
    mapping.insert(
        "International_cricketers_of_South_African_origin".to_owned(),
        (17416221, Some(258)),
    );
    mapping.insert("Independence_Day_(film)".to_owned(), (52389, Some(105387)));
    mapping.insert(
        "Camelot,_Chesapeake,_Virginia".to_owned(),
        (91342, Some(49222)),
    );
    mapping.insert("SBC_Communications".to_owned(), (26213969, Some(444015)));
    mapping.insert("Superman_(film)".to_owned(), (28381, Some(79015)));
    mapping.insert("Rabobank_(cycling_team)".to_owned(), (2354465, Some(6233)));
    mapping.insert("U._Chandana".to_owned(), (896434, Some(3520028)));
    mapping.insert("LPGA_Championship".to_owned(), (229059, Some(281917)));
    mapping.insert(
        "Hapoel_Be'er_Sheva_A.F.C.".to_owned(),
        (5834903, Some(986529)),
    );
    let reader = File::open(&args.input_wiki2qid).unwrap();
    for record in Reader::new(reader).unwrap() {
        let record = from_value::<MappingRecord>(&record.unwrap()).unwrap();

        if titles.contains(&record.title) {
            mapping
                .try_insert(record.title, (record.pageid, record.qid))
                .ok();
        }
    }

    let train = generate_dataset(train, &mapping);
    let validation = generate_dataset(validation, &mapping);
    let test = generate_dataset(test, &mapping);

    write_dataset(
        train,
        Path::new(&args.output_dir)
            .join("train.parquet")
            .to_str()
            .unwrap(),
    );
    write_dataset(
        validation,
        Path::new(&args.output_dir)
            .join("validation.parquet")
            .to_str()
            .unwrap(),
    );
    write_dataset(
        test,
        Path::new(&args.output_dir)
            .join("test.parquet")
            .to_str()
            .unwrap(),
    );
}
