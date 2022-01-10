"""Test text2lists."""
from pathlib import Path
from radiobee.loadtext import loadtext
from radiobee.text2lists import text2lists


def test_text2lists():
    """Test text2lists data\test-dual.txt."""
    filename = r"data\test-dual.txt"
    text = loadtext(filename)  # noqa
    l1, l2 = text2lists(text)
    assert l2[0] in [""]
    assert "国际\n中\n双语" in l1[0]


def test_shakespeare1000():
    """Separate first 1000.
    
    from pathlib import Path
    import zipfile
    dir_loc = r""
    filename = r"莎士比亚 - 莎士比亚全集（套装共39本 英汉双语）-外语教学与研究出版社 (2016).txt.zip"
    zfile = zipfile.ZipFile(Path(dir_loc) / filename)
    res_bytes = zfile.read(zfile.infolist()[0])
    encoding = cchardet.detect(res_bytes).get("encoding")

    text1000 = []
    line = 0
    numb_lines = 4000
    for elm in res_bytes.splitlines():
        if elm.decode(encoding).strip():
            text1000.append(elm.decode(encoding))
            if line >= numb_lines - 1:
                break
            line += 1
    Path(f"data/shakespeare-zh-en-{numb_lines}.txt").write_text("\n".join(text1000), encoding="utf8")
    
    tset = cmat2test(cmat)
    df = pd.DataFrame(tset).rename(columns=dict(zip(range(0, 3), ['x', 'y', 'cos'])))
    plot_df(df)
    
    """
    # text1000a = Path("data/shakespeare-zh-en-1000.txt").read_text(encoding="utf8")
    # text2000 = Path("data/shakespeare-zh-en-1000.txt").read_text(encoding="utf8")
    text4000 = Path("data/shakespeare-zh-en-4000.txt").read_text(encoding="utf8")

    # l1000a, l10002b = text2lists(text1000)
    # l2000a, l2000b = text2lists(text2000)
    
    l4000, r4000 = text2lists(text4000)
