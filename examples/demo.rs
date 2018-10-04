// A bunch of this file is lifted from https://github.com/Twinklebear/tobj_viewer


#[macro_use]
extern crate glium;

use glium::{glutin, Surface, Display, vertex::VertexBufferAny};
use std::{
  f32,
  path::Path,
};

use cgmath::{
  Rad,
  vec3 as v3,
};
type M4 = cgmath::Matrix4<f32>;
type P3 = cgmath::Point3<f32>;
type V3 = cgmath::Vector3<f32>;

pub fn load_wavefront(display: &Display, path: &Path) -> (VertexBufferAny, f32) {
  #[derive(Copy, Clone)]
  struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color_diffuse: [f32; 3],
    color_specular: [f32; 4],
  }

  implement_vertex!(Vertex, position, normal, color_diffuse, color_specular);

  let mut min_pos = [f32::INFINITY; 3];
  let mut max_pos = [f32::NEG_INFINITY; 3];
  let mut vertex_data = Vec::new();
  match tobj::load_obj(path) {
    Ok((models, mats)) => {
      println!("{:?}", mats);
      // Just upload the first object in the group
      for model in &models {
        println!("{:?}", model);
        let mesh = &model.mesh;
        println!("Uploading model: {}", model.name);
        for idx in &mesh.indices {
          let i = *idx as usize;
          let pos = [
            mesh.positions[3 * i],
            mesh.positions[3 * i + 1],
            mesh.positions[3 * i + 2],
          ];
          let normal = if !mesh.normals.is_empty() {
            [
              mesh.normals[3 * i],
              mesh.normals[3 * i + 1],
              mesh.normals[3 * i + 2],
            ]
          } else {
            [0.0, 0.0, 0.0]
          };
          let (color_diffuse, color_specular) = match mesh.material_id {
            Some(i) => (
              mats[i].diffuse,
              [
                mats[i].specular[0],
                mats[i].specular[1],
                mats[i].specular[2],
                mats[i].shininess,
              ],
            ),
            None => ([0.8, 0.8, 0.8], [0.15, 0.15, 0.15, 15.0]),
          };
          println!("{:?}", color_diffuse);

          vertex_data.push(Vertex {
            position: pos,
            normal: normal,
            color_diffuse: color_diffuse,
            color_specular: color_specular,
          });
          // Update our min/max pos so we can figure out the bounding box of the object
          // to view it
          for i in 0..3 {
            min_pos[i] = f32::min(min_pos[i], pos[i]);
            max_pos[i] = f32::max(max_pos[i], pos[i]);
          }
        }
      }
    }
    Err(e) => panic!("Loading of {:?} failed due to {:?}", path, e),
  }
  // Compute scale factor to fit the model with a [-1, 1] bounding box
  let diagonal_len = 6.0;
  let current_len = f32::powf(max_pos[0] - min_pos[0], 2.0)
    + f32::powf(max_pos[1] - min_pos[1], 2.0)
    + f32::powf(max_pos[2] - min_pos[2], 2.0);
  let scale = f32::sqrt(diagonal_len / current_len);
  println!("Model scaled by {} to fit", scale);
  (
    glium::vertex::VertexBuffer::new(display, &vertex_data)
      .unwrap()
      .into_vertex_buffer_any(),
    scale,
  )
}

fn main() {
  let mut events_loop = glutin::EventsLoop::new();
  let window = glutin::WindowBuilder::new();
  let context = glutin::ContextBuilder::new();
  let display = glium::Display::new(window, context, &events_loop).unwrap();


  let (vertex_buffer, _scale) = load_wavefront(&display, Path::new(&"examples/streets/Street_Straight.obj"));

  let program = program!(&display,
                         140 => {
                           vertex: "
                               #version 140
                               uniform mat4 proj;
                               uniform mat4 view;
                               uniform mat4 model;
                               in vec3 position;
                               in vec3 normal;
                               in vec3 color_diffuse;
                               out vec3 v_position;
                               out vec3 v_normal;
                               out vec3 v_color;
                               void main() {
                                   v_position = position;
                                   v_normal = normal;
                                   v_color = color_diffuse;
                                   gl_Position = proj * view * model * vec4(v_position, 1.0);
                               }
                           ",

                           fragment: "
                               #version 140
                               in vec3 v_normal;
                               in vec3 v_color;
                               out vec4 f_color;
                               const vec3 LIGHT = vec3(-0.2, 0.8, 0.1);
                               void main() {
                                   float lum = max(dot(normalize(v_normal), normalize(LIGHT)), 0.0);
                                   vec3 color = (0.3 + 0.7 * lum) * v_color;
                                   f_color = vec4(color, 1.0);
                               }
                           ",
                         },
  ).unwrap();

  let draw = || {
    let view = M4::look_at(
      P3::new(-1.0, -1.0, 1.0),
      P3::new(0.0, 0.0, 0.0),
      v3(0.0, 0.0, 1.0),
    );

    let proj: M4 = cgmath::ortho::<f32>(-5.0, 5.0,
                                        -5.0, 5.0,
                                        -5.0, 5.0);


    let params = glium::DrawParameters {
      depth: glium::Depth {
        test: glium::DepthTest::IfLess,
        write: true,
        ..Default::default()
      },
      ..Default::default()
    };



    // drawing a frame
    let mut target = display.draw();
    target.clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);

    let size = 1;

    for x in 0..size {
      for y in 0..size {
        let model =
          M4::from_translation(v3(x as f32, y as f32, 0.0)) * M4::from_angle_x(Rad(std::f32::consts::FRAC_PI_2));

        let uniforms = uniform! {
          view: cgmath::conv::array4x4(view),
          proj: cgmath::conv::array4x4(proj),
          model: cgmath::conv::array4x4(model),
        };

        target.draw(&vertex_buffer,
                    &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
                    &program,
                    &uniforms,
                    &params).unwrap();

      }
    }




    target.finish().unwrap();
  };

  events_loop.run_forever(|event| {
    match event {
      glutin::Event::WindowEvent { event, .. } => match event {
        // Break from the main loop when the window is closed.
        glutin::WindowEvent::CloseRequested => return glutin::ControlFlow::Break,
        // Redraw the triangle when the window is resized.
        glutin::WindowEvent::Resized(..) => draw(),
        _ => (),
      },
      _ => (),
    }
    glutin::ControlFlow::Continue
  });
}
