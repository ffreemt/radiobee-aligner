"""Test shuffle_sents.

    eps: float = 6
    min_samples: int = 4
    tf_type: str = "linear"
    idf_type: Optional[str] = None
    dl_type: Optional[str] = None
    norm: Optional[str] = None
    lang1: Optional[str] = "en"
    lang2: Optional[str] = "zh"
"""
from radiobee.seg_text import seg_text
from radiobee.shuffle_sents import shuffle_sents
from radiobee.align_sents import align_sents

text1 = """`Wretched inmates!' I ejaculated mentally, `you deserve perpetual isolation from your species for your churlish inhospitality. At least, I would not keep my doors barred in the day time. I don't care--I will get in!' So resolved, I grasped the latch and shook it vehemently. Vinegar-faced Joseph projected his head from a round window of the barn."""
text2 = """“被囚禁的囚犯!”我在精神上被射精,“你应该永远与你的物种隔绝,因为你这种粗鲁的病态。至少,我白天不会锁门,我不在乎,我进去了!”我决心如此,我抓住了门锁,狠狠地摇了一下。醋脸的约瑟夫从谷仓的圆窗朝他的头照射。"""
text3 = """"Elende Insassen! ejakulierte ich im Geiste, "ihr verdient die ewige Isolation von eurer Spezies für eure rüpelhafte Ungastlichkeit. Zumindest würde ich meine Türen tagsüber nicht verriegeln. Das ist mir egal - ich werde reinkommen!' So entschlossen, ergriff ich die Klinke und rüttelte heftig daran. Der essiggesichtige Joseph streckte seinen Kopf aus einem runden Fenster der Scheune."""


def test_shuffle_sents_en_zh():
    """Test shuffle_sents_en_zh."""
    sents_en = seg_text(text1)
    sents_zh = seg_text(text2)

    lang1 = "en"
    lang2 = "zh"

    pairs = shuffle_sents(sents_en, sents_zh)
    pairs_ = shuffle_sents(sents_en, sents_zh, lang1=lang1, lang2=lang2)

    # pairs[3] == ('', "I don't care--I will get in!'", '')
    assert pairs == pairs_

    # assert not pairs[3][0]
    # after swapping
    assert not pairs[3][1]


def test_shuffle_sents_en_de():
    """Test shuffle_sents_en_de."""
    sents_en = seg_text(text1)
    sents_de = seg_text(text3)

    lang1 = "en"
    lang2 = "de"

    pairs = shuffle_sents(sents_en, sents_de)
    pairs_ = shuffle_sents(sents_en, sents_de, lang1=lang1, lang2=lang2)

    assert pairs == pairs_

    #
    # assert not pairs[3][0]
    _ = """In [218]: pairs[:2]
    Out[218]:
    [["`Wretched inmates!'", '', ''],
     ['I ejaculated mentally, `you deserve perpetual isolation from your species for your churlish inhospitality.',
      '"Elende Insassen! ejakulierte ich im Geiste, "ihr verdient die ewige Isolation von eurer Spezies für eure rüpelhafte Ungastlichkeit.',
      0.62]]
    """
    assert not pairs[0][1]
    assert "mentally" in str(pairs[1]) and "Elende" in str(pairs[1])

    # [elm[2] for elm in pairs]
    # ['', 0.62, 0.72, 0.74, 0.68, 0.79]
    if isinstance(pairs[1][2], float):
        assert pairs[1][2] > 0.6
    if isinstance(pairs[2][2], float):
        assert pairs[2][2] > 0.7
    if isinstance(pairs[3][2], float):
        assert pairs[3][2] > 0.7
    if isinstance(pairs[4][2], float):
        assert pairs[4][2] > 0.6
    if isinstance(pairs[5][2], float):
        assert pairs[5][2] > 0.7


_ = """
In [232]: shuffle_sents.cmat.round(2)
Out[232]:
array([[ 0.27,  0.62,  0.07,  0.11,  0.02,  0.02],
       [ 0.03,  0.09,  0.72,  0.18,  0.07, -0.07],
       [ 0.19,  0.07,  0.16,  0.74, -0.01, -0.02],
       [-0.02,  0.18,  0.16,  0.06,  0.68, -0.04],
       [ 0.02,  0.07,  0.04, -0.04,  0.02,  0.79]], dtype=float32)
pairs[1]
sents_en[1], sents_de[0], shuffle_sents.cmat[0, 1]
['I ejaculated mentally, `you deserve perpetual isolation from your species for your churlish inhospitality.',
 '"Elende Insassen! ejakulierte ich im Geiste, "ihr verdient die ewige Isolation von eurer Spezies für eure rüpelhafte Ungastlichkeit.',
 0.62]

pairs[2]
sents_en[2], sents_de[1], shuffle_sents.cmat[1, 2].round(2)
Out[244]:
('At least, I would not keep my doors barred in the day time.',
 'Zumindest würde ich meine Türen tagsüber nicht verriegeln.',
 0.72)
...

import mtplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
set_style("darkgrind")
plt.ion()

ali = shuffle_sents(sents_en, sents_de)
sns.heatmap(shuffle_sents.cmat, cmap="viridis_r").invert_yaxis()
ax = plt.gca()
ax.set_xlabel(shuffle_sents.lang1)
ax.set_ylabel(shuffle_sents.lang2)

ali == [["`Wretched inmates!'", '', ''],
 ['I ejaculated mentally, `you deserve perpetual isolation from your species for your churlish inhospitality.',
  '"Elende Insassen! ejakulierte ich im Geiste, "ihr verdient die ewige Isolation von eurer Spezies für eure rüpelhafte Ungastlichkeit.',
  0.62],
 ['At least, I would not keep my doors barred in the day time.',
  'Zumindest würde ich meine Türen tagsüber nicht verriegeln.',
  0.72],
 ["I don't care--I will get in!'",
  "Das ist mir egal - ich werde reinkommen!'",
  0.74],
 ['So resolved, I grasped the latch and shook it vehemently.',
  'So entschlossen, ergriff ich die Klinke und rüttelte heftig daran.',
  0.68],
 ['Vinegar-faced Joseph projected his head from a round window of the barn.',
  'Der essiggesichtige Joseph streckte seinen Kopf aus einem runden Fenster der Scheune.',
  0.79]]

res1 = align_sents(sents_en, sents_de)
ali = shuffle_sents(sents_en, sents_de)
for idx in range(1, 6):
    assert res1[idx] == tuple(ali[idx][:2])
"""
