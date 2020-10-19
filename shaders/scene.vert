#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 0) out vec2 v_TexCoord;
layout(push_constant) uniform framentPushConstants {
    float zoom;
    float pad;
    vec2 offset;
    int camera_index;
} pc;
layout(set = 0, binding = 0) buffer Locals {
  mat4 u_Transform[];
};
void main() {
  v_TexCoord = a_TexCoord;
  gl_Position = u_Transform[pc.camera_index] * a_Pos;
}    

