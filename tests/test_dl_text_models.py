"""DL-B·DL-C 모델(사전 문서 E8)의 구조 계약 — 어긋나면 지연·확률이 다른 모델의 것이 된다."""
import torch

import text_models
import train as T


def test_charcnn_tiny_matches_prereg_structure():
    net = text_models.build_model("charcnn_tiny", 4)
    assert net.embedding.embedding_dim == 32
    assert [(c.kernel_size[0], c.out_channels) for c in net.convs] == [(3, 64)]


def test_charcnn_k3_matches_dl_c_front_size():
    # E9-1: t_k3 = t_front 로 두는 근거 — 보조 헤드 경로와 같은 모양이어야 한다.
    net, ee = text_models.build_model("charcnn_k3", 4), text_models.build_model("charcnn_ee", 4)
    assert net.embedding.embedding_dim == ee.embedding.embedding_dim
    assert [(c.kernel_size[0], c.out_channels) for c in net.convs] == [(3, ee.convs[0].out_channels)]
    assert net.classifier.weight.shape == ee.aux_head.weight.shape


def test_charcnn_ee_heads_follow_train_and_eval_contract():
    torch.manual_seed(0)
    net = text_models.build_model("charcnn_ee", 4)
    x = torch.randint(0, 257, (5, 40))
    net.train()
    out = net(x)
    assert isinstance(out, list) and len(out) == 2 and out[0].shape == (5, 4)
    net.eval()
    with torch.no_grad():
        aux, main = net.forward_heads(x)
        assert torch.equal(net(x), main)
        # 보조 헤드는 커널 3 분기만 본다 — latency_fair.charcnn_parts 의 DL-C 앞부분 비용과 같은 연산.
        pooled3 = torch.relu(net.convs[0](net.embedding(x).transpose(1, 2))).max(dim=2).values
        assert torch.allclose(aux, net.aux_head(pooled3))
    assert "charcnn_ee" in T.MULTI_HEAD_MODELS and {"charcnn_tiny", "charcnn_k3", "charcnn_ee"} <= T.TEXT_MODELS


def test_distilbert_trims_padding_without_changing_logits():
    # E10: 배치를 실제 최대 길이로 잘라도 결과가 같아야 512 토큰 설정이 계산만 아낀다.
    from transformers import DistilBertConfig
    import cross_validate as CV

    torch.manual_seed(0)
    cfg = DistilBertConfig(vocab_size=50, dim=16, n_layers=1, n_heads=2, hidden_dim=32,
                           max_position_embeddings=64)
    net = text_models.build_model("distilbert", 4, config=cfg).eval()
    x = torch.zeros(3, 64, dtype=torch.long)
    x[0, :10], x[1, :5], x[2, :7] = 7, 9, 11
    with torch.no_grad():
        full = net.bert(input_ids=x, attention_mask=(x != 0).long()).logits
        assert torch.allclose(net(x), full, atol=1e-5)
    assert "distilbert" in CV.TOKEN_MODELS and "distilbert" not in T.TEXT_MODELS
