use dynamo_parsers_v2::{
    UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState,
    create_unified_parser_for_family,
};
use serde_json::{Value, json};
fn run(input: &str, cuts: &[usize]) -> Value {
    let mut p = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
    p.initialize_request(UnifiedParserInit {
        starting_state: UnifiedParserStartingState::None,
        ..Default::default()
    })
    .unwrap();
    let mut out = UnifiedParserOutput::default();
    let mut from = 0;
    for &to in cuts {
        if let Err(e) = p.parse_into(&input[from..to], &mut out) {
            return json!({"error":e.to_string(),"stage":"push","events":out.assembled()});
        }
        from = to;
    }
    match p.finish() {
        Ok(mut tail) => {
            out.append(&mut tail);
            json!({"events":out.assembled()})
        }
        Err(e) => json!({"error":e.to_string(),"stage":"finish","events":out.assembled()}),
    }
}
fn main() {
    let cases=vec![
        ("valid_xml", "<｜DSML｜ calls><｜DSML｜ invoke name=\"inspect\"><｜DSML｜ parameter name=\"value\" string=\"true\">Paris</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>".to_string()),
        ("valid_json", "<｜DSML｜ calls><｜DSML｜ invoke name=\"inspect\">{\"value\":\"Paris\"}</｜DSML｜ invoke></｜DSML｜ calls>".to_string()),
        ("json_embedded_markers", format!("<｜DSML｜ calls><｜DSML｜ invoke name=\"inspect\">{}</｜DSML｜ invoke><｜DSML｜ invoke name=\"done\">{{}}</｜DSML｜ invoke></｜DSML｜ calls>",json!({"value":" café 🐈 </｜DSML｜ invoke> </｜DSML｜ calls> \\\"\n","nested":{"values":[true,null,42,-1250]}}))),
        ("invalid_json", "<｜DSML｜ calls><｜DSML｜ invoke name=\"inspect\">{\"value\":oops}</｜DSML｜ invoke></｜DSML｜ calls>".to_string()),
        ("xml_missing_string", "<｜DSML｜ calls><｜DSML｜ invoke name=\"inspect\"><｜DSML｜ parameter name=\"value\">Paris</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>".to_string()),
        ("xml_bad_string", "<｜DSML｜ calls><｜DSML｜ invoke name=\"inspect\"><｜DSML｜ parameter name=\"value\" string=\"maybe\">Paris</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>".to_string()),
        ("xml_trailing_junk", "<｜DSML｜ calls><｜DSML｜ invoke name=\"inspect\"><｜DSML｜ parameter name=\"value\" string=\"true\">Paris</｜DSML｜ parameter>garbage</｜DSML｜ invoke></｜DSML｜ calls>".to_string()),
        ("truncated_xml", "<｜DSML｜ calls><｜DSML｜ invoke name=\"inspect\"><｜DSML｜ parameter name=\"value\" string=\"true\">unfinished".to_string()),
        ("plain_text", "Hello world".to_string()),
    ];
    for (name, input) in cases {
        let boundaries: Vec<usize> = (0..=input.len())
            .filter(|i| input.is_char_boundary(*i))
            .collect();
        let reference = run(&input, &[input.len()]);
        let mut partition_matches = 0;
        let mut errors = 0;
        let mut sample = Value::Null;
        for &split in &boundaries {
            let result = run(&input, &[split, input.len()]);
            if result.get("error") == reference.get("error")
                && result.get("events") == reference.get("events")
            {
                partition_matches += 1;
            }
            if result.get("error").is_some() {
                errors += 1;
            }
            if sample.is_null() || result.get("error").is_some() {
                sample = result;
            }
        }
        let chars = run(&input, &boundaries[1..]);
        println!(
            "{}",
            json!({"case":name,"utf8_split_trials":boundaries.len(),"split_errors":errors,"partition_matches":partition_matches,"single_character":chars,"sample":sample})
        );
    }
}
