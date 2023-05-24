#version 460

layout(location = 0) in vec4 ClipPos;
layout(location = 1) in vec4 PrevClipPos;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec2 FragMotion;

void main() {
    FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    FragMotion = PrevClipPos.xy / PrevClipPos.w - ClipPos.xy / ClipPos.w;
}