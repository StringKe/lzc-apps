import os
import shutil
from pathlib import Path
from typing import List, Optional

from git import Repo
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class DocsVectorCreator:
    def __init__(
        self,
        repo_url: str,
        branch: str = "main",
        docs_dir: str = "docs",
        file_filter: str = "**/*.md",
    ):
        """
        初始化文档向量化工具

        Args:
            repo_url: Git 仓库地址
            branch: Git 分支名称
            docs_dir: 文档所在目录
            file_filter: 文件过滤器模式
        """
        self.repo_url = repo_url
        self.branch = branch
        self.docs_dir = docs_dir
        self.file_filter = file_filter
        self.temp_dir = Path("temp_repo")
        self.db_path = Path("vectorstore")

    def clone_repository(self) -> None:
        """克隆指定的 Git 仓库"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)
        Repo.clone_from(self.repo_url, self.temp_dir, branch=self.branch)

    def load_documents(self) -> List[Document]:
        """加载文档"""
        docs_path = self.temp_dir / self.docs_dir
        if not docs_path.exists():
            raise ValueError(f"文档目录 {docs_path} 不存在")

        # 使用 GitLoader 加载文档
        loader = GitLoader(
            repo_path=str(self.temp_dir),
            branch=self.branch,
            file_filter=lambda file_path: (
                file_path.startswith(self.docs_dir) and file_path.endswith(".md")
            ),
        )

        return loader.load()

    def create_vector_store(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Chroma:
        """
        创建向量存储

        Args:
            documents: 文档列表
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
        """
        # 分割文档
        text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(documents)

        # 创建向量存储
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings(),
            persist_directory=str(self.db_path),
        )
        vector_store.persist()
        return vector_store

    def cleanup(self) -> None:
        """清理临时文件"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def process(self, api_key: Optional[str] = None) -> None:
        """
        处理文档向量化的主流程

        Args:
            api_key: OpenAI API 密钥
        """
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        try:
            print("开始克隆仓库...")
            self.clone_repository()

            print("加载文档...")
            documents = self.load_documents()
            print(f"共加载了 {len(documents)} 个文档")

            print("创建向量存储...")
            vector_store = self.create_vector_store(documents)
            print(f"向量存储已创建完成，保存在 {self.db_path} 目录")

        finally:
            self.cleanup()


if __name__ == "__main__":
    creator = DocsVectorCreator(
        repo_url="https://gitee.com/lazycatcloud/lzc-developer-doc.git",
        branch="main",
        docs_dir="docs",
    )

    # 替换为你的 OpenAI API 密钥
    creator.process(api_key="your-api-key-here")
