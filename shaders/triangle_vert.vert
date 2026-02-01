#version 450

layout( push_constant ) uniform constants
{
    float frame;
} PushConstants;

const vec3 positions[3] = {
    vec3(0.f,  1.f,  0.0f),
    vec3(1.f,  -1.f, 0.0f),
    vec3(-1.f, -1.f, 0.0f),
};

void main()
{
    vec3 aPos = positions[gl_VertexIndex];

    float a = PushConstants.frame * 3.141592 / 4.;
    mat4 rot = mat4(cos(a), -sin(a), 0., 0.,
                    sin(a),  cos(a), 0., 0.,
                        0.,      0., 1., 0.,
                        0.,      0., 0., 1.);
    vec4 pos = vec4(aPos.x, aPos.y, aPos.z, 1.0);
    pos = vec4(0.1,0.1,0.1,1.0) * pos;
    pos += vec4(0.5,0.5,0.0,0.0);
    gl_Position = rot * pos;
    gl_Position.y *= -1.0f; // Emulate GL
}
