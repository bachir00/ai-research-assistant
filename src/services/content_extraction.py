"""
Service d'extraction de contenu web.
Supporte HTML, PDF et autres formats de documents.
"""

import aiohttp
import asyncio
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from urllib.parse import urljoin, urlparse
from datetime import datetime
import re
import mimetypes

from asyncssh import logger

from src.core.logging import setup_logger
from src.models.document_models import Document, DocumentType

# Import conditionnel des dépendances
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    PyPDF2 = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


class ContentExtractionError(Exception):
    """Exception pour les erreurs d'extraction de contenu."""
    pass


class WebContentExtractor:
    """
    Extracteur de contenu web avec support multi-format.
    """
    
    def __init__(self, timeout: int = 30, max_content_length: int = 10_000_000):
        self.logger = setup_logger("content_extractor")
        self.timeout = timeout
        self.max_content_length = max_content_length
        
        # Headers pour simuler un navigateur réel
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Vérification des dépendances
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Vérifie que les dépendances nécessaires sont installées."""
        if not BEAUTIFULSOUP_AVAILABLE:
            self.logger.warning("BeautifulSoup4 non installé - extraction HTML limitée")
        if not PDF_AVAILABLE:
            self.logger.warning("PyPDF2 non installé - extraction PDF non disponible")
        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests non installé - extraction synchrone non disponible")
    
    async def extract_content(self, url: str) -> Document:
        """
        Extrait le contenu d'une URL.
        
        Args:
            url: URL à extraire
            
        Returns:
            Document avec le contenu extrait
            
        Raises:
            ContentExtractionError: Si l'extraction échoue
        """
        self.logger.info(f"Extraction de contenu: {url}")
        
        try:
            # Détecter le type de contenu
            content_type = await self._detect_content_type(url)
            
            if content_type.startswith('application/pdf'):
                return await self._extract_pdf_content(url)
            elif content_type.startswith('text/html') or 'html' in content_type:
                return await self._extract_html_content(url)
            else:
                # Tentative d'extraction générique
                #################### faire aussi l'extraction en fonction de l'extension du fichier et le js ####################
                return await self._extract_generic_content(url)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction de {url}: {str(e)}")
            raise ContentExtractionError(f"Impossible d'extraire le contenu de {url}: {str(e)}")
    
    async def _detect_content_type(self, url: str) -> str:
        """Détecte le type de contenu d'une URL."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.head(url, headers=self.headers) as response:
                    content_type = response.headers.get('content-type', '').lower()
                    if content_type:
                        return content_type.split(';')[0]  # Enlever le charset
                    
            # Fallback: détecter par extension
            parsed_url = urlparse(url)
            content_type, _ = mimetypes.guess_type(parsed_url.path)
            return content_type or 'text/html'
            
        except Exception as e:
            self.logger.warning(f"Impossible de détecter le type de contenu pour {url}: {e}")
            return 'text/html'  # Default fallback
    
    async def _extract_html_content(self, url: str) -> Document:
        """Extrait le contenu d'une page HTML."""
        if not BEAUTIFULSOUP_AVAILABLE:
            raise ContentExtractionError("BeautifulSoup4 non installé pour l'extraction HTML")
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    raise ContentExtractionError(f"Erreur HTTP {response.status} pour {url}")
                
                # Vérifier la taille du contenu
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_content_length:
                    raise ContentExtractionError(f"Contenu trop volumineux: {content_length} bytes")
                
                html_content = await response.text()
        
        # Parser avec BeautifulSoup
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extraire le titre
        title = self._extract_title(soup)
        
        # Extraire le contenu principal
        content = self._extract_main_content(soup)
        # Vérifier la longueur du contenu
        if len(content) > self.max_content_length:
            raise ContentExtractionError(f"Contenu extrait trop volumineux: {len(content)} caractères")
        # Afficher le contenu
        # self.logger.info(f"Contenu extrait ({len(content)} caractères)")
        
        # Extraire les métadonnées
        author = self._extract_author(soup)
        publish_date = self._extract_publish_date(soup)
        
        return Document(
            title=title,
            url=url,
            content=content,
            doc_type=DocumentType.ARTICLE,
            author=author,
            published_date=publish_date,
            word_count=len(content.split()),
            language='fr'  ############################################# Détection automatique à implémenter ###################
        )
    
    def _extract_title(self, soup: "BeautifulSoup") -> str:
        """Extrait le titre de la page."""
        # Priorité: title tag, h1, og:title, première heading
        
        # Title tag
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text().strip():
            return title_tag.get_text().strip()
        
        # Meta og:title
        og_title = soup.find('meta', {'property': 'og:title'})
        if og_title and og_title.get('content'):
            return og_title.get('content').strip()
        
        # Premier h1
        h1 = soup.find('h1')
        if h1 and h1.get_text().strip():
            return h1.get_text().strip()
        
        # Fallback
        return "Titre non trouvé"
    
    def _extract_main_content(self, soup: "BeautifulSoup") -> str:
        """Extrait le contenu principal de la page."""
        # Supprimer les éléments indésirables
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
            element.decompose()
        
        # Supprimer les commentaires
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
            comment.extract()
        
        # Chercher le contenu principal dans l'ordre de priorité
        content_selectors = [
            'article',
            '[role="main"]',
            'main',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '#content',
            '.main-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element
                break
        
        # Fallback: tout le body
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extraire le texte en gardant la structure
        return self._clean_text(main_content.get_text())
    
    def _clean_text(self, text: str) -> str:
        """Nettoie et formate le texte extrait."""
        if not text:
            return ""
        
        # Supprimer les espaces multiples et les sauts de ligne excessifs
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Supprimer les espaces en début et fin
        text = text.strip()
        
        # Limiter la longueur si nécessaire
        if len(text) > 50000:  # 50k caractères max
            text = text[:50000] + "... [Contenu tronqué]"
        
        return text
    
    def _extract_author(self, soup: "BeautifulSoup") -> Optional[str]:
        """Extrait l'auteur de l'article."""
        # Meta author
        author_meta = soup.find('meta', {'name': 'author'})
        if author_meta and author_meta.get('content'):
            return author_meta.get('content').strip()
        
        # Schema.org author
        author_schema = soup.find(attrs={'itemprop': 'author'})
        if author_schema:
            return author_schema.get_text().strip()
        
        # Recherche par classe CSS commune
        author_selectors = [
            '.author',
            '.byline',
            '.post-author',
            '.article-author'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author_text = element.get_text().strip()
                if author_text and len(author_text) < 100:  # Limite raisonnable
                    return author_text
        
        return None
    
    def _extract_publish_date(self, soup: "BeautifulSoup") -> Optional[datetime]:
        """Extrait la date de publication."""
        # Meta published_time
        time_meta = soup.find('meta', {'property': 'article:published_time'})
        if time_meta and time_meta.get('content'):
            try:
                from dateutil.parser import parse
                return parse(time_meta.get('content'))
            except:
                pass
        
        # Schema.org datePublished
        date_schema = soup.find(attrs={'itemprop': 'datePublished'})
        if date_schema:
            date_str = date_schema.get('datetime') or date_schema.get_text()
            try:
                from dateutil.parser import parse
                return parse(date_str)
            except:
                pass
        
        return None
    
    async def _extract_pdf_content(self, url: str) -> Document:
        """Extrait le contenu d'un PDF."""
        if not PDF_AVAILABLE:
            raise ContentExtractionError("PyPDF2 non installé pour l'extraction PDF")
        
        # Télécharger le PDF
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    raise ContentExtractionError(f"Erreur HTTP {response.status} pour {url}")
                
                pdf_content = await response.read()
        
        # Extraire le texte du PDF
        try:
            import io
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            title = "Document PDF"
            content = ""
            
            # Extraire le texte de toutes les pages
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                content += page_text + "\n"
            
            # Nettoyer le contenu
            content = self._clean_text(content)
            
            return Document(
                title=title,
                url=url,
                content=content,
                doc_type=DocumentType.ACADEMIC_PAPER,
                word_count=len(content.split()),
                language='fr'  ############################################# Détection automatique à implémenter ###################
            )
            
        except Exception as e:
            raise ContentExtractionError(f"Erreur lors de l'extraction PDF: {str(e)}")
    
    async def _extract_generic_content(self, url: str) -> Document:
        """Extraction générique pour les autres types de contenu."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    raise ContentExtractionError(f"Erreur HTTP {response.status} pour {url}")
                
                content = await response.text()
        
        # Nettoyage basique
        content = self._clean_text(content)
        
        return Document(
            title=f"Document depuis {urlparse(url).netloc}",
            url=url,
            content=content,
            doc_type=DocumentType.OTHER,
            word_count=len(content.split()),
            language='fr'
        )


class ContentExtractionManager:
    """
    Gestionnaire d'extraction de contenu avec gestion des erreurs et retry.
    """
    
    def __init__(self, max_concurrent: int = 5, max_retries: int = 2):
        self.logger = setup_logger("extraction_manager")
        self.extractor = WebContentExtractor()
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def extract_multiple(self, urls: List[str]) -> List[Document]:
        """
        Extrait le contenu de plusieurs URLs en parallèle.
        
        Args:
            urls: Liste des URLs à extraire
            
        Returns:
            Liste des documents extraits (peut contenir moins d'éléments en cas d'erreur)
        """
        self.logger.info(f"Extraction de contenu pour {len(urls)} URLs")
        
        # Créer les tâches d'extraction
        tasks = [self._extract_with_retry(url) for url in urls]
        
        # Exécuter en parallèle avec limite de concurrence
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les résultats valides
        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Document):
                documents.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Échec d'extraction pour {urls[i]}: {str(result)}")
            else:
                self.logger.warning(f"Résultat inattendu pour {urls[i]}: {type(result)}")
        
        self.logger.info(f"Extraction terminée: {len(documents)}/{len(urls)} succès")
        return documents
    
    async def _extract_with_retry(self, url: str) -> Document:
        """Extrait le contenu d'une URL avec retry automatique."""
        async with self.semaphore:
            last_error = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    if attempt > 0:
                        # Attendre entre les tentatives
                        await asyncio.sleep(2 ** attempt)
                        self.logger.info(f"Tentative {attempt + 1}/{self.max_retries + 1} pour {url}")
                    
                    return await self.extractor.extract_content(url)
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries:
                        self.logger.warning(f"Tentative {attempt + 1} échouée pour {url}: {str(e)}")
                    else:
                        self.logger.error(f"Toutes les tentatives ont échoué pour {url}: {str(e)}")
            
            # Si toutes les tentatives échouent
            raise last_error or ContentExtractionError(f"Échec d'extraction pour {url}")
        




##########################################################""        
# Exemple d'utilisation (à exécuter dans un contexte asynchrone)
async def main():
    extractor_manager = ContentExtractionManager(max_concurrent=3, max_retries=2)
    urls = [
        'https://www.iana.org/help/example-domains',
        'https://documents1.worldbank.org/curated/en/691261636143890139/pdf/Taxing-Pollution.pdf'
    ]
    documents = await extractor_manager.extract_multiple(urls)
    for doc in documents:
        print(f"Title: {doc.title}, URL: {doc.url}, Word Count: {doc.word_count}, Language: {doc.language}, Content Length: {len(doc.content)}, \nContenu tronqué: {doc.content[:500]}")
        logger.error("⚠️   pytest n'est pas installé. Impossible de tester les erreurs de validation.")

if __name__ == "__main__":
    asyncio.run(main())