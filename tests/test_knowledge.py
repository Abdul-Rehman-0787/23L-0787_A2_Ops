from app.knowledge import makeChunks, readSections, splitText


def testReadSections(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# My title\n\nIntro text\n\n## First\nAlpha\n\n## Second\nBeta\n", encoding="utf-8")
    title, sections = readSections(str(path))
    assert title == "My title"
    assert sections == [("Overview", "Intro text"), ("First", "Alpha"), ("Second", "Beta")]


def testSplitTextKeepsParagraphsTogether():
    assert splitText("aaa\n\nbbb\n\nccc", 8) == ["aaa\n\nbbb", "ccc"]


def testMakeChunksSkipsEmptySectionsAndOtherFiles(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n## One\nText one\n\n## Empty\n\n## Two\nText two\n",
                                   encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    chunks = makeChunks(str(tmp_path))
    assert [chunk["section"] for chunk in chunks] == ["One", "Two"]
    assert chunks[0]["source"] == "a.md"
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)