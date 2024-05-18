import torch

# 저장된 모델 파일 경로 설정
model_path = 'saved_model/best_model_e10_b128_l0.01.pth'  # 예시 파일명, 실제 경로에 맞게 조정하세요

# 모델 정보 로드
model_info = torch.load(model_path)

# valid_loss 값 출력
print(f"Loaded model from fold {model_info['fold']} and epoch {model_info['epoch']} with validation loss: {model_info['valid_loss']}")
