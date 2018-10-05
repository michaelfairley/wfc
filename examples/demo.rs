// A bunch of this file is lifted from https://github.com/Twinklebear/tobj_viewer
#[macro_use]
extern crate glium;

use glium::{glutin, Surface, Display, vertex::VertexBufferAny};
use std::{
  f32,
  path::Path,
};

use wfc::{
  Rotation,
};

use cgmath::{
  Rad,
  vec3 as v3,
};
type M4 = cgmath::Matrix4<f32>;
type P3 = cgmath::Point3<f32>;

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
      // println!("{:?}", mats);
      // Just upload the first object in the group
      for model in &models {
        // println!("{:?}", model);
        let mesh = &model.mesh;
        // println!("Uploading model: {}", model.name);
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
          // println!("{:?}", color_diffuse);

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
  // println!("Model scaled by {} to fit", scale);
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

  let (width, height) = display.get_framebuffer_dimensions();
  let mut aspect = width as f32 / height as f32;

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
                                   // TODO: inverse transpose
                                   v_normal = (model * vec4(normal, 0.0)).xyz;
                                   v_color = color_diffuse;
                                   gl_Position = proj * view * model * vec4(v_position, 1.0);
                               }
                           ",

                           fragment: "
                               #version 140
                               in vec3 v_normal;
                               in vec3 v_color;
                               out vec4 f_color;
                               const vec3 LIGHT = vec3(0.2, 0.3, 0.9);
                               void main() {
                                   float lum = max(dot(normalize(v_normal), normalize(LIGHT)), 0.0);
                                   vec3 color = (0.5 + 0.5 * lum) * v_color;
                                   f_color = vec4(color, 1.0);
                               }
                           ",
                         },
  ).unwrap();

  let models = vec![
    load_wavefront(&display, Path::new(&"examples/streets/Street_Straight.obj")).0,
    load_wavefront(&display, Path::new(&"examples/streets/Street_Curve.obj")).0,
  ];

  let variants = vec![
    (0, Rotation::None),
    (0, Rotation::R90),
    (1, Rotation::None),
    (1, Rotation::R90),
    (1, Rotation::R180),
    (1, Rotation::R270),
  ];

  let street_top = [0, 2, 5];
  let blank_top = [1, 3, 4];

  let street_right = [1, 2, 3];
  let blank_right = [0, 4, 5];

  let street_bottom = [0, 3, 4];
  let blank_bottom = [1, 2, 5];

  let street_left = [1, 4, 5];
  let blank_left = [0, 2, 3];

  // top, right, bottom, left
  let neighbors = vec![
    //Straight
    [street_bottom, blank_left, street_
  ];

  let size = 2;

  // let mut adjacencies = HashMap::new();

  let gen = || {
    wfc::generate(
      (size, size),
      variants.len(),
      // edges,
    )
  };

  let mut cells = gen();

  // let cells = [
  //   Cell(0, Rotation::None), Cell(0, Rotation::R90),
  //   Cell(0, Rotation::None), Cell(1, Rotation::R270),
  // ];

  let mut closed = false;

  while !closed {
    events_loop.poll_events(|event| {
      match event {
        glutin::Event::WindowEvent { event, .. } => match event {
          glutin::WindowEvent::CloseRequested => { closed = true; return; },
          glutin::WindowEvent::KeyboardInput{input: glutin::KeyboardInput{ virtual_keycode:Some(glutin::VirtualKeyCode::Escape), .. }, .. } => { closed = true; return; }
          glutin::WindowEvent::KeyboardInput{input: glutin::KeyboardInput{ virtual_keycode:Some(glutin::VirtualKeyCode::R), state: glutin::ElementState::Pressed, .. }, .. } => { cells = gen(); }
          glutin::WindowEvent::Resized(..) => {
            let (width, height) = display.get_framebuffer_dimensions();
            aspect = width as f32 / height as f32;
          }
          _ => (),
        },
        _ => (),
      }
    });

    let view = M4::look_at(
      P3::new(-1.0, -1.0, 1.0),
      P3::new(0.0, 0.0, 0.0),
      v3(0.0, 0.0, 1.0),
    );

    let ss = size as f32 * 1.2;

    let proj: M4 = cgmath::ortho::<f32>(-ss * aspect / 2.0, ss * aspect / 2.0,
                                        -1.0, ss - 1.0,
                                        0.0, 100.0);


    let params = glium::DrawParameters {
      depth: glium::Depth {
        test: glium::DepthTest::IfLess,
        write: true,
        ..Default::default()
      },
      ..Default::default()
    };



    let mut target = display.draw();
    target.clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);

    for x in 0..size {
      for y in 0..size {
        let cell = variants[cells[y * size + x]];

        let (modeli, rotation) = cell;

        let rot = match rotation {
          Rotation::None => M4::from_scale(1.0),
          Rotation::R90 => M4::from_angle_z(Rad(std::f32::consts::FRAC_PI_2 * 3.0)),
          Rotation::R180 => M4::from_angle_z(Rad(std::f32::consts::PI)),
          Rotation::R270 => M4::from_angle_z(Rad(std::f32::consts::FRAC_PI_2)),
        };

        let model =
          M4::from_translation(v3(x as f32, y as f32, 0.0))
          * rot
          * M4::from_angle_x(Rad(std::f32::consts::FRAC_PI_2))
          * M4::from_scale(0.5);



        let mesh = &models[modeli];

        let uniforms = uniform! {
          view: cgmath::conv::array4x4(view),
          proj: cgmath::conv::array4x4(proj),
          model: cgmath::conv::array4x4(model),
        };

        target.draw(mesh,
                    &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
                    &program,
                    &uniforms,
                    &params).unwrap();

      }
    }

    target.finish().unwrap();


  }
}
