use color_eyre::Result;
use context_attribute::context;
use filtering::low_pass_filter::LowPassFilter;
use framework::{AdditionalOutput, MainOutput};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct VelocityFilter {
    pub filtered_velocity: LowPassFilter<f32>,
    pub filtered_acceleration: LowPassFilter<f32>,
}

#[context]
pub struct CreationContext {
    velocity_low_pass_filter_coefficient:
        Parameter<f32, "velocity_filter.velocity_low_pass_filter_coefficient">,
    acceleration_low_pass_filter_coefficient:
        Parameter<f32, "velocity_filter.acceleration_low_pass_filter_coefficient">,
}

#[context]
pub struct CycleContext {
    filtered_velocity: AdditionalOutput<f32, "filtered_velocity">,
    filtered_acceleration: AdditionalOutput<f32, "filtered_acceleration">,

    last_step_velocity: Input<Option<f32>, "last_step_velocity?">,
}

#[context]
#[derive(Default)]
pub struct MainOutputs {
    pub filtered_velocity: MainOutput<f32>,
    pub filtered_acceleration: MainOutput<f32>,
}

impl VelocityFilter {
    pub fn new(context: CreationContext) -> Result<Self> {
        Ok(VelocityFilter {
            filtered_velocity: LowPassFilter::with_smoothing_factor(
                0.0,
                *context.velocity_low_pass_filter_coefficient,
            ),
            filtered_acceleration: LowPassFilter::with_smoothing_factor(
                0.0,
                *context.acceleration_low_pass_filter_coefficient,
            ),
        })
    }

    pub fn cycle(&mut self, mut context: CycleContext) -> Result<MainOutputs> {
        self.filtered_velocity
            .update(*context.last_step_velocity.unwrap_or(&0.0));

        context
            .filtered_velocity
            .fill_if_subscribed(|| self.filtered_velocity.state());
        context
            .filtered_acceleration
            .fill_if_subscribed(|| self.filtered_acceleration.state());

        Ok(MainOutputs {
            filtered_velocity: self.filtered_velocity.state().into(),
            filtered_acceleration: self.filtered_acceleration.state().into(),
        })
    }
}
