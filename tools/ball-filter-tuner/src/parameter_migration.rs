//! Explicit import of parameter files written before literal limit semantics.
//! Never run this on an already converted configuration: a new zero is literal.
use color_eyre::{Result, eyre::eyre};
use serde_json::{Value, json};

pub fn from_legacy(value: &mut Value) -> Result<()> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| eyre!("ball-filter parameters must be an object"))?;
    for (key, upper) in [
        ("maximum_detection_distance", 1000.0),
        ("maximum_matching_distance", 1000.0),
        ("reacquisition_matching_distance", 1000.0),
        ("radius_consistency_maximum_distance", 1000.0),
        ("publication_maximum_distance", 1000.0),
        ("selection_confidence_cap", 1_000_000.0),
        ("publication_maximum_covariance_ratio", 1_000_000.0),
        ("field_boundary_confidence_decay_distance", 1_000_000.0),
    ] {
        if object
            .get(key)
            .and_then(Value::as_f64)
            .is_some_and(|x| x <= 0.0)
            || (key == "publication_maximum_covariance_ratio" && !object.contains_key(key))
        {
            object.insert(key.into(), json!(upper));
        }
    }
    if object
        .get("maximum_detection_radius_ratio")
        .and_then(Value::as_f64)
        .is_some_and(|x| x <= 1.0)
    {
        object.insert("maximum_detection_radius_ratio".into(), json!(1_000_000.0));
    }
    if object
        .get("publication_detection_noise")
        .and_then(Value::as_f64)
        .is_some_and(|x| x <= 0.0)
    {
        object.insert("publication_detection_noise".into(), json!(0.05));
    }
    for key in [
        "publication_maximum_age",
        "visible_missed_detection_timeout",
        "near_visible_missed_detection_timeout",
    ] {
        if object
            .get(key)
            .is_some_and(|x| x["secs"] == 0 && x["nanos"] == 0)
        {
            object.insert(key.into(), json!({"secs":1_000_000,"nanos":0}));
            // Previously a disabled near timeout also suppressed its independent
            // decay rate. Preserve that behavior explicitly when importing.
            if key == "near_visible_missed_detection_timeout"
                && object
                    .get("near_visible_missed_validity_decay_rate")
                    .is_some_and(|x| !x.is_null())
            {
                object.insert("near_visible_missed_validity_decay_rate".into(), json!(0.0));
            }
        }
    }
    object.remove("maximum_matching_cost_validity_penalty_factor");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_conversion_changes_sentinels_but_preserves_literal_zero_weights() {
        let mut value = json!({"maximum_matching_distance":0.0,"maximum_detection_radius_ratio":1.0,
            "publication_maximum_age":{"secs":0,"nanos":0},"association_uncertainty_weight":0.0,
            "hidden_validity_decay_rate":0.0,"resting_velocity_threshold":0.0,
            "near_visible_missed_detection_timeout":{"secs":0,"nanos":0},
            "near_visible_missed_validity_decay_rate":20.0,
            "maximum_matching_cost_validity_penalty_factor":0.14});
        from_legacy(&mut value).unwrap();
        assert_eq!(value["maximum_matching_distance"], 1000.0);
        assert_eq!(value["maximum_detection_radius_ratio"], 1_000_000.0);
        assert_eq!(value["publication_maximum_age"]["secs"], 1_000_000);
        assert_eq!(value["publication_maximum_covariance_ratio"], 1_000_000.0);
        for key in [
            "association_uncertainty_weight",
            "hidden_validity_decay_rate",
            "resting_velocity_threshold",
            "near_visible_missed_validity_decay_rate",
        ] {
            assert_eq!(value[key], 0.0);
        }
        assert!(
            value
                .get("maximum_matching_cost_validity_penalty_factor")
                .is_none()
        );
    }
}
