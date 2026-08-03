# รายงาน ARC3 Sovereign Pure NumPy v0.41.0

โปรเจกต์เดียวสำหรับการแข่งขัน ใช้ neural inference ด้วย NumPy ล้วนบน CPU และรวมข้อมูลที่โครงการสร้างทั้งหมด 692,000 records ไว้ครั้งเดียว ไม่มี PyTorch, JAX, TensorFlow, vLLM, Transformers, Qwen, CUDA model runtime หรือ native extension ใน submission path

## การฝึกและอำนาจตัดสินใจ
- action/causal/static heads ใช้ข้อมูลฝึกรวม 575,537 records
- NumPy ensemble เป็น proposal-only; legality, transaction freshness, deadline และ official outcome ยังเป็น authoritative
- static prior ถูกกักกันและไม่มีสิทธิ์เลือก action เพราะ held-out generalization ต่ำ

## หลักฐานภายใน
- pytest: 158 tests, 158 passed, 0 failed
- procedural test macro: 39.2556% → 40.4952%
- worst split: 18.2800% → 18.3200%
- illegal predictions: 0
- causal test accuracy: moved 95.6832%, surprise 82.0652%, repair 81.0559%, conflict 63.0901%
- latency 1,000 decisions: median 1.789 ms, p95 2.007 ms, p99 2.412 ms
- deterministic wheel, deterministic Kaggle bundle, clean-room no-index import, source manifest, ZIP CRC และ post-extraction verification: PASS

ทั้งหมดเป็น synthetic/procedural evidence ไม่ใช่คะแนนการแข่งขันจริง สถานะทางการยัง UNVERIFIED จนกว่าจะผ่าน committed private Kaggle rerun
