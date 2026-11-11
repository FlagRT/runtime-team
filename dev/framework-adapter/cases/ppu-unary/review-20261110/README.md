# 本轮实验资格核对

2026-11-10执行者Codex逐项核对qual-1110-02原始result.json：SiLU/sigmoid各三精度、shape32×64，每组3次，共18条；前反向对PPU原生及CPU FP32参考全部在预设容差内，每次GEMS前反向各命中1次，dispatcher恢复一致。原始结果SHA为27fff36b…，完整值在records.json。

按该证据显式填写6条实验passed记录，供独立routes测试使用。不是环境探测后自动批准，也不扩大到模型、其他形状、混合精度自动转换或其他编译器版本。记录加载时核对原始证据SHA；其他平台旧记录不可复用。
