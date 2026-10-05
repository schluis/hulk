//! Timestamped, read-only images of production-filter replay on simulator recordings.
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::Path,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use ball_filter::{BallMode, tracker::Tracker};
use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use projection::Projection;
use serde_json::{Value, json};
use types::parameters::BallFilterParameters;

use crate::{
    ReferenceFrame,
    recording::{Recording, Reference},
    scoring,
};

/// Each frame uses the output-cycle Ground frame. Camera data carries its separate
/// exposure timestamp; its projected hypotheses are taken at that exposure.
pub struct Clip {
    name: String,
    frames: Vec<Value>,
    parameters: BallFilterParameters,
}

impl Clip {
    pub fn read(
        path: &Path,
        capture: &BallFilterParameters,
        parameters: &BallFilterParameters,
    ) -> Result<Self> {
        let recording = Recording::read(
            path,
            "",
            "simulation/ball_ground_truth",
            ReferenceFrame::Ground,
        )?;
        scoring::verify(&recording, capture)?;
        let start = recording
            .cycles
            .first()
            .ok_or_else(|| eyre!("empty recording"))?
            .time
            .as_nanos();
        let poses: std::collections::BTreeMap<_, _> = recording
            .cycles
            .iter()
            .filter_map(|cycle| cycle.ground_to_field.map(|pose| (cycle.time, pose)))
            .collect();
        let mut tracker = Tracker::default();
        let mut frames = Vec::new();
        let mut previous_truth: Option<(i64, nalgebra::Point3<f32>)> = None;
        for cycle in &recording.cycles {
            let mut camera_frame = None;
            let mut obstacle_snapshot = None;
            for input in &cycle.inputs {
                let percepts = tracker.advance_with_obstacles(
                    input.time,
                    input.odometry,
                    input.detections.as_deref(),
                    input.camera.as_ref(),
                    input.obstacles.as_ref(),
                    parameters,
                    &cycle.dimensions,
                )?;
                if let Some(detections) = &input.detections {
                    obstacle_snapshot = input.obstacles.clone();
                    let camera = input.camera.as_ref().filter(|camera| {
                        ball_filter::tracker::camera_is_recent(
                            input.time,
                            camera.time,
                            parameters.maximum_camera_matrix_age,
                        )
                    });
                    let projected: Vec<_> = tracker
                        .filter
                        .hypotheses
                        .iter()
                        .enumerate()
                        .filter_map(|(index, h)| {
                            let camera = &camera?.inner;
                            let pixel = camera
                                .ground_with_z_to_pixel(
                                    h.position().position,
                                    cycle.dimensions.ball_radius,
                                )
                                .ok()?;
                            Some(json!({"index": index, "pixel": [pixel.x(), pixel.y()]}))
                        })
                        .collect();
                    let boxes: Vec<_> = detections.iter().map(|o| json!({
                        "label": format!("{:?}", o.label), "confidence": o.bounding_box.confidence,
                        "min": [o.bounding_box.area.min.x(), o.bounding_box.area.min.y()],
                        "max": [o.bounding_box.area.max.x(), o.bounding_box.area.max.y()]
                    })).collect();
                    camera_frame = Some(json!({
                        "time_ns": input.time.as_nanos(), "camera_time_ns": camera.map(|c| c.time.as_nanos()),
                        "size": camera.map(|c| [c.inner.image_size.x(), c.inner.image_size.y()]),
                        "detections": boxes, "hypotheses": projected,
                        "percepts": percepts, "obstacles": input.obstacles,
                    }));
                }
            }
            let estimate = tracker.finish_with_field_pose(
                cycle.time,
                parameters,
                &cycle.dimensions,
                cycle.field_prior_pose,
            );
            let truth = match &cycle.reference {
                Some(Reference::Ground(points)) => {
                    Some(points.iter().map(|p| [p.x(), p.y()]).collect::<Vec<_>>())
                }
                _ => None,
            };
            let field_truth = cycle
                .motion_reference
                .as_ref()
                .filter(|points| points.len() == 1)
                .map(|p| p[0]);
            let velocity = field_truth
                .zip(previous_truth)
                .and_then(|(point, (time, previous))| {
                    let dt = (cycle.time.as_nanos() - time) as f32 * 1e-9;
                    if !(dt > 0.0 && dt <= 0.1) {
                        return None;
                    }
                    let velocity: nalgebra::Vector2<f32> = (point.inner - previous).xy() / dt;
                    if !velocity.iter().all(|x| x.is_finite()) || velocity.norm() > 15.0 {
                        return None;
                    }
                    let rotation = cycle.ground_to_field?.inner.rotation.inverse();
                    let ground = rotation * velocity;
                    Some([ground.x, ground.y])
                });
            previous_truth = field_truth.map(|p| (cycle.time.as_nanos(), p.inner));
            if camera_frame.is_none() {
                continue;
            }
            // Match a source pose within 20ms, like the 3D viewer. Report its
            // timestamp; never silently use today's pose for old observations.
            let mut obstacle_pose_time = None;
            let obstacles = obstacle_snapshot.as_ref().and_then(|snapshot| {
                let (stamp, pose) = [poses.range(..=snapshot.time).next_back(), poses.range(snapshot.time..).next()]
                    .into_iter().flatten().min_by_key(|(stamp, _)| stamp.as_nanos().abs_diff(snapshot.time.as_nanos()))
                    .filter(|(stamp, _)| stamp.as_nanos().abs_diff(snapshot.time.as_nanos()) <= 20_000_000)?;
                obstacle_pose_time = Some(stamp.as_nanos());
                let old_to_current = cycle.ground_to_field?.inverse() * *pose;
                Some(snapshot.inner.iter().filter(|o| o.kind == types::obstacles::ObstacleKind::Robot).map(|obstacle| {
                    let p = old_to_current * obstacle.position;
                    json!({"position":[p.x(),p.y()],"radius":obstacle.radius_at_foot_height.max(obstacle.radius_at_hip_height)})
                }).collect::<Vec<_>>())
            });
            let selected = tracker.filter.best_hypothesis_with_field_pose(
                parameters,
                &cycle.dimensions,
                cycle.field_prior_pose,
            );
            let hypotheses: Vec<_> = tracker.filter.hypotheses.iter().enumerate().map(|(index, h)| {
                let ball = h.position();
                json!({"index": index, "position": [ball.position.x(), ball.position.y()],
                    "velocity": [ball.velocity.x(), ball.velocity.y()], "validity": h.validity,
                    "mode": match h.mode { BallMode::Moving(_) => "moving", BallMode::Resting(_) => "resting" },
                    "selected": selected.is_some_and(|s| std::ptr::eq(s, h)),
                    "age_seconds": (cycle.time.as_nanos() - h.last_seen.as_nanos()) as f64 * 1e-9,
                    "covariance": h.position_covariance(), "state": h})
            }).collect();
            frames.push(json!({"time_ns": cycle.time.as_nanos(), "seconds": (cycle.time.as_nanos() - start) as f64 * 1e-9,
                "truth": truth, "truth_velocity": velocity, "hypotheses": hypotheses,
                "estimate": estimate.map(|e| json!({"position": [e.position.x(),e.position.y()],"velocity":[e.velocity.x(),e.velocity.y()]})),
                "camera": camera_frame, "ground_to_field": cycle.ground_to_field,
                "field_prior_pose": cycle.field_prior_pose, "field_dimensions": cycle.dimensions,
                "obstacle_pose_time_ns": obstacle_pose_time, "obstacles": obstacles, "obstacle_time_ns": obstacle_snapshot.as_ref().map(|o|o.time.as_nanos()),
            }));
        }
        ensure!(!frames.is_empty(), "recording contains no detection frames");
        Ok(Self {
            name: path.display().to_string(),
            frames,
            parameters: parameters.clone(),
        })
    }
}

/// Bounded by the supplied recordings; no background rendering or unbounded frame queue.
pub struct ImageServer {
    clips: Arc<RwLock<Vec<Clip>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    pub address: SocketAddr,
}
impl ImageServer {
    pub fn start(address: SocketAddr) -> Result<Self> {
        ensure!(
            address.ip().is_loopback(),
            "image interface must bind to loopback; use SSH forwarding for remote access"
        );
        let listener = TcpListener::bind(address)?;
        let address = listener.local_addr()?;
        listener.set_nonblocking(true)?;
        let clips = Arc::new(RwLock::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let data = clips.clone();
        let stopping = stop.clone();
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        if let Err(error) = respond(&mut stream, &data) {
                            eprintln!("Image request: {error}");
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(20))
                    }
                    Err(error) => {
                        eprintln!("Image listener: {error}");
                        break;
                    }
                }
            }
        });
        eprintln!("Ball-filter images: http://{address}");
        Ok(Self {
            clips,
            stop,
            thread: Some(thread),
            address,
        })
    }
    pub fn add(&self, clip: Clip) {
        self.clips.write().expect("image clips lock").push(clip);
    }
}
impl Drop for ImageServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn respond(stream: &mut TcpStream, clips: &RwLock<Vec<Clip>>) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let mut buffer = [0_u8; 8192];
    let mut used = 0;
    let deadline = Instant::now() + Duration::from_secs(2);
    while used < buffer.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        ensure!(!remaining.is_zero(), "request header timeout");
        stream.set_read_timeout(Some(remaining))?;
        let n = stream.read(&mut buffer[used..])?;
        if n == 0 {
            break;
        }
        used += n;
        if buffer[..used].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    let request = std::str::from_utf8(&buffer[..used])?;
    let mut words = request.lines().next().unwrap_or("").split_whitespace();
    let method = words.next().unwrap_or("");
    let target = words.next().unwrap_or("");
    let (status, content_type, body) = if method != "GET" {
        (
            "405 Method Not Allowed",
            "text/plain",
            "GET required".into(),
        )
    } else {
        route(target, &clips.read().expect("image clips lock"))
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    Ok(())
}

fn route(target: &str, clips: &[Clip]) -> (&'static str, &'static str, String) {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let query: std::collections::BTreeMap<_, _> =
        query.split('&').filter_map(|p| p.split_once('=')).collect();
    if path == "/" {
        return (
            "200 OK",
            "text/html; charset=utf-8",
            include_str!("observation.html").into(),
        );
    }
    if path == "/clips.json" {
        return ("200 OK", "application/json", json!(clips.iter().enumerate().map(|(i,c)| json!({
            "id":i,"name":c.name,"frames":c.frames.len(),"duration":c.frames.last().unwrap()["seconds"],
        })).collect::<Vec<_>>()).to_string());
    }
    if !matches!(
        path,
        "/frame.json" | "/frame.svg" | "/frames.json" | "/parameters.json"
    ) {
        return ("404 Not Found", "text/plain", "unknown endpoint".into());
    }
    let clip = query
        .get("clip")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let Some(clip) = clips.get(clip) else {
        return (
            "404 Not Found",
            "text/plain",
            "no such clip; wait for the first demo scenario to finish".into(),
        );
    };
    if path == "/frames.json" {
        return (
            "200 OK",
            "application/json",
            serde_json::to_string(&clip.frames).unwrap(),
        );
    }
    if path == "/parameters.json" {
        return (
            "200 OK",
            "application/json",
            serde_json::to_string(&clip.parameters).unwrap(),
        );
    }
    let index = if let Some(seconds) = query.get("time") {
        let Ok(seconds) = seconds.parse::<f64>() else {
            return ("400 Bad Request", "text/plain", "invalid time".into());
        };
        if !seconds.is_finite() || seconds < 0.0 {
            return ("400 Bad Request", "text/plain", "invalid time".into());
        }
        clip.frames
            .partition_point(|f| f["seconds"].as_f64().unwrap() < seconds)
            .min(clip.frames.len() - 1)
    } else {
        query
            .get("index")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    };
    let Some(frame) = clip.frames.get(index) else {
        return ("404 Not Found", "text/plain", "no such frame".into());
    };
    if path == "/frame.svg" {
        ("200 OK", "image/svg+xml", render(frame))
    } else {
        (
            "200 OK",
            "application/json",
            json!({"clip":clip.name,"index":index,"frame":frame}).to_string(),
        )
    }
}

fn number(v: &Value) -> f64 {
    v.as_f64().filter(|v| v.is_finite()).unwrap_or(0.0)
}
fn xy(v: &Value) -> (f64, f64) {
    (number(&v[0]), number(&v[1]))
}
fn render(frame: &Value) -> String {
    use std::fmt::Write;
    let mut svg = String::from(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="840" viewBox="0 0 1280 840"><rect width="1280" height="840" fill="#101b28"/><style>text{font-family:monospace;fill:#dce6f0;font-size:14px}.small{font-size:12px}</style><defs><clipPath id="field"><rect x="20" y="80" width="600" height="600"/></clipPath><clipPath id="camera"><rect x="650" y="80" width="610" height="360"/></clipPath></defs>"##,
    );
    write!(svg, r#"<text x="20" y="28" font-size="20">Ball filter · {:.3}s · output timestamp {} ns</text>"#, number(&frame["seconds"]), frame["time_ns"]).unwrap();
    svg.push_str(r##"<text x="20" y="56">Ground view: forward ↑, left ← · grid 1m · arrows = 0.3s travel</text><rect x="20" y="80" width="600" height="600" fill="#17392e"/><g clip-path="url(#field)">"##);
    for i in -4..=4 {
        let p = 320 + i * 75;
        write!(
            svg,
            r##"<path d="M{p} 80V680 M20 {}H620" stroke="#416053" stroke-width="1"/>"##,
            p + 60
        )
        .unwrap();
    }
    svg.push_str(r##"<circle cx="320" cy="380" r="75" fill="none" stroke="#d6dbca" stroke-dasharray="5 5"/><path d="M320 366L311 394L329 394Z" fill="#e9edf2"/>"##);
    let point = |v: &Value| {
        let (x, y) = xy(v);
        (320.0 - y * 75.0, 380.0 - x * 75.0)
    };
    if let Some(obstacles) = frame["obstacles"].as_array() {
        for obstacle in obstacles {
            let (x, y) = point(&obstacle["position"]);
            write!(svg,r##"<circle cx="{x}" cy="{y}" r="{}" fill="#865334" fill-opacity="0.55" stroke="#ff9865" stroke-dasharray="4 3"/>"##,number(&obstacle["radius"])*75.0).unwrap();
        }
    }
    if let Some(truth) = frame["truth"].as_array() {
        for ball in truth {
            let (x, y) = point(ball);
            write!(svg,r##"<circle cx="{x}" cy="{y}" r="8" fill="#58e7a1"/><text x="{}" y="{}">truth</text>"##,x+10.0,y-10.0).unwrap();
            if frame["truth_velocity"].is_array() {
                arrow(&mut svg, x, y, &frame["truth_velocity"], "#58e7a1");
            }
        }
    }
    for h in frame["hypotheses"].as_array().unwrap() {
        let (x, y) = point(&h["position"]);
        let color = if h["selected"] == true {
            "#50baff"
        } else {
            "#c9b9ef"
        };
        write!(svg,r#"<circle cx="{x}" cy="{y}" r="6" fill="none" stroke="{color}" stroke-width="2"/><text x="{}" y="{}">h{}</text>"#,x+8.0,y+16.0,h["index"]).unwrap();
        arrow(&mut svg, x, y, &h["velocity"], color);
    }
    if let Some(e) = frame.get("estimate").filter(|v| v.is_object()) {
        let (x, y) = point(&e["position"]);
        write!(svg,r##"<rect x="{}" y="{}" width="18" height="18" fill="none" stroke="#ffd06b" stroke-width="2"/>"##,x-9.0,y-9.0).unwrap();
        arrow(&mut svg, x, y, &e["velocity"], "#ffd06b");
    }
    svg.push_str("</g>");
    if let (Some(now), Some(stamp), Some(pose_stamp)) = (
        frame["time_ns"].as_i64(),
        frame["obstacle_time_ns"].as_i64(),
        frame["obstacle_pose_time_ns"].as_i64(),
    ) {
        write!(svg,r#"<text x="28" y="668" class="small">Orange robots: age {:.0}ms; source-pose offset {:+.0}ms</text>"#,(now-stamp) as f64*1e-6,(pose_stamp-stamp) as f64*1e-6).unwrap();
    }
    svg.push_str(r##"<text x="650" y="56">Camera projection (synthetic; no RGB)</text><rect x="650" y="80" width="610" height="360" fill="#243346"/><g clip-path="url(#camera)">"##);
    let camera = &frame["camera"];
    let (w, h) = xy(&camera["size"]);
    if w > 0.0 && h > 0.0 {
        // Uniform scale preserves the image aspect ratio.
        let scale = (610.0 / w).min(360.0 / h);
        write!(svg, r##"<defs><clipPath id="sensor"><rect x="650" y="80" width="{}" height="{}"/></clipPath></defs><rect x="650" y="80" width="{}" height="{}" fill="#182635" stroke="#7690a6"/><g clip-path="url(#sensor)">"##,w*scale,h*scale,w*scale,h*scale).unwrap();
        for detection in camera["detections"].as_array().unwrap() {
            let (x, y) = xy(&detection["min"]);
            let (right, bottom) = xy(&detection["max"]);
            let color = if detection["label"] == "Ball" {
                "#ffd06b"
            } else {
                "#ff9865"
            };
            write!(svg,r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="{color}" stroke-width="2"/>"#,650.0+x*scale,80.0+y*scale,(right-x)*scale,(bottom-y)*scale).unwrap();
        }
        for hypothesis in camera["hypotheses"].as_array().unwrap() {
            let (x, y) = xy(&hypothesis["pixel"]);
            write!(svg,r##"<circle cx="{}" cy="{}" r="5" fill="none" stroke="#c9b9ef"/><text x="{}" y="{}">c{}</text>"##,650.0+x*scale,80.0+y*scale,657.0+x*scale,80.0+y*scale,hypothesis["index"]).unwrap();
        }
    }
    if w > 0.0 && h > 0.0 {
        svg.push_str("</g>");
    }
    svg.push_str("</g>");
    write!(
        svg,
        r#"<text x="650" y="465" class="small">Exposure: {} ns; indices are frame-local</text>"#,
        camera["time_ns"]
    )
    .unwrap();
    svg.push_str(
        r#"<text x="650" y="491">Hypotheses at output time (position m; speed m/s)</text>"#,
    );
    for (row, h) in frame["hypotheses"]
        .as_array()
        .unwrap()
        .iter()
        .take(12)
        .enumerate()
    {
        let (x, y) = xy(&h["position"]);
        let (vx, vy) = xy(&h["velocity"]);
        write!(svg,r#"<text x="650" y="{}" class="small">h{} {} ({x:.2}, {y:.2}) v={:.2} age={:.2}s confidence={:.2}{}</text>"#,515+row*22,h["index"],h["mode"].as_str().unwrap(),vx.hypot(vy),number(&h["age_seconds"]),number(&h["validity"]),if h["selected"]==true {" selected"} else {""}).unwrap();
    }
    svg.push_str(r##"<text x="20" y="715" fill="#58e7a1">Green: truth</text><text x="20" y="740">Blue: selected primary · purple: other hypotheses</text><text x="20" y="765">Gold square/arrow: published estimate (may differ)</text><text x="20" y="790">Dashed circle: 1m kicking region · camera boxes: ball gold, robot orange</text><text x="20" y="815">h/c indices are local to their own output/exposure frame, not persistent IDs.</text></svg>"##);
    svg
}
fn arrow(svg: &mut String, x: f64, y: f64, velocity: &Value, color: &str) {
    use std::fmt::Write;
    let (vx, vy) = xy(velocity);
    let dx = -vy * 22.5;
    let dy = -vx * 22.5;
    let length = dx.hypot(dy);
    if length < 0.5 {
        return;
    }
    let (ex, ey) = (x + dx, y + dy);
    let (ux, uy) = (dx / length, dy / length);
    write!(svg,r#"<path d="M{x} {y}L{ex} {ey} M{} {}L{ex} {ey}L{} {}" fill="none" stroke="{color}" stroke-width="2"/>"#,ex-ux*7.0-uy*4.0,ey-uy*7.0+ux*4.0,ex-ux*7.0+uy*4.0,ey-uy*7.0-ux*4.0).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    fn clip() -> Clip {
        let frame = |seconds: f64| {
            json!({"seconds":seconds,"time_ns":(seconds*1e9) as i64,
            "truth":null,"truth_velocity":null,"hypotheses":[],"estimate":null,
            "camera":{"time_ns":0,"size":null,"hypotheses":[],"detections":[]}})
        };
        Clip {
            name: "example.mcap".into(),
            frames: vec![frame(0.0), frame(0.04), frame(0.08)],
            parameters: json5::from_str(include_str!(
                "../../../etc/parameters/base/ball_filter.json5"
            ))
            .unwrap(),
        }
    }
    #[test]
    fn seeks_first_frame_at_or_after_time_and_rejects_invalid_times() {
        let clips = vec![clip()];
        let (_, _, body) = route("/frame.json?time=0.03", &clips);
        assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["index"], 1);
        for time in ["NaN", "inf", "-1", "no"] {
            assert_eq!(
                route(&format!("/frame.json?time={time}"), &clips).0,
                "400 Bad Request"
            );
        }
        assert_eq!(route("/frame.json?index=99", &clips).0, "404 Not Found");
        assert_eq!(route("/frame.svg?clip=99", &clips).0, "404 Not Found");
        let (_, kind, svg) = route("/frame.svg?index=1", &clips);
        assert_eq!(kind, "image/svg+xml");
        assert!(svg.contains("no RGB") && svg.contains("0.040s"));
    }
    #[test]
    fn serves_images_over_http_and_stops_with_owner() {
        let server = ImageServer::start("127.0.0.1:0".parse().unwrap()).unwrap();
        server.add(clip());
        let mut stream = TcpStream::connect(server.address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        stream
            .write_all(b"GET /frame.svg?time=0.04 HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("Content-Type: image/svg+xml") && response.contains("0.040s"));
        let address = server.address;
        drop(server);
        assert!(TcpStream::connect(address).is_err());
    }
}
