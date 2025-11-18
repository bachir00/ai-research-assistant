"""
Service de chunking pour la gestion des textes longs.
Divise intelligemment les documents en chunks pour le traitement par LLM.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.core.logging import setup_logger


@dataclass
class TextChunk:
    """Représente un chunk de texte avec métadonnées."""
    content: str
    start_index: int
    end_index: int
    chunk_id: int
    total_chunks: int
    word_count: int
    has_heading: bool = False
    heading_text: Optional[str] = None


class TextChunker:
    """
    Service de découpage intelligent de texte pour le traitement par LLM.
    
    Fonctionnalités:
    - Découpage respectant les phrases et paragraphes
    - Préservation des titres et structure
    - Gestion du chevauchement entre chunks
    - Optimisation pour les limites de tokens LLM
    """
    
    def __init__(
        self,
        max_chunk_size: int = 4000,  # En caractères
        overlap_size: int = 200,     # Chevauchement entre chunks
        min_chunk_size: int = 500    # Taille minimale d'un chunk
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.logger = setup_logger("text_chunker")
        
        # Patterns pour identifier la structure
        self.heading_patterns = [
            r'^#{1,6}\s+.+$',           # Markdown headings
            r'^\d+\.\s+.+$',            # Numérotations
            r'^[A-Z\s]{5,}$',           # Titres en majuscules
            r'^\w+:$',                  # Labels avec deux-points
        ]
        
        self.sentence_endings = r'[.!?]+(?:\s|$)'
        self.paragraph_breaks = r'\n\s*\n'
    
    def chunk_text(self, text: str, preserve_structure: bool = True) -> List[TextChunk]:
        """
        Découpe un texte en chunks intelligents.
        
        Args:
            text: Texte à découper
            preserve_structure: Préserver la structure (titres, paragraphes)
            
        Returns:
            Liste des chunks créés
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Nettoyage préliminaire
        text = self._clean_text(text)
        
        # Si le texte est assez court, retourner un seul chunk
        if len(text) <= self.max_chunk_size:
            return [TextChunk(
                content=text,
                start_index=0,
                end_index=len(text),
                chunk_id=1,
                total_chunks=1,
                word_count=len(text.split())
            )]
        
        # Découpage intelligent
        if preserve_structure:
            chunks = self._chunk_with_structure(text)
        else:
            chunks = self._chunk_simple(text)
        
        # Post-traitement des chunks
        chunks = self._post_process_chunks(chunks)
        
        self.logger.info(f"Texte découpé en {len(chunks)} chunks (taille moyenne: {sum(len(c.content) for c in chunks) // len(chunks)} caractères)")
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Nettoie le texte avant découpage."""
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normaliser les sauts de ligne
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Supprimer les espaces en début et fin
        text = text.strip()
        
        return text
    
    def _chunk_with_structure(self, text: str) -> List[TextChunk]:
        """Découpage en préservant la structure du document."""
        chunks = []
        current_chunk = ""
        current_start = 0
        
        # Diviser en paragraphes
        paragraphs = re.split(self.paragraph_breaks, text)
        text_position = 0
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # Vérifier si le paragraphe contient un titre
            is_heading, heading_text = self._detect_heading(paragraph)
            
            # Si ajouter ce paragraphe dépasse la limite
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                # Sauvegarder le chunk actuel
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    current_start,
                    text_position,
                    len(chunks) + 1
                )
                chunks.append(chunk)
                
                # Commencer un nouveau chunk avec chevauchement
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + paragraph
                current_start = text_position - len(overlap_text)
            else:
                # Ajouter le paragraphe au chunk actuel
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    current_start = text_position
            
            text_position += len(paragraph) + 2  # +2 pour \n\n
        
        # Ajouter le dernier chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                current_start,
                len(text),
                len(chunks) + 1
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_simple(self, text: str) -> List[TextChunk]:
        """Découpage simple par phrases."""
        chunks = []
        sentences = re.split(self.sentence_endings, text)
        
        current_chunk = ""
        current_start = 0
        text_position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Estimer la position dans le texte original
            sentence_in_text = sentence + "."  # Approximation
            
            if len(current_chunk) + len(sentence_in_text) > self.max_chunk_size and current_chunk:
                # Sauvegarder le chunk actuel
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    current_start,
                    text_position,
                    len(chunks) + 1
                )
                chunks.append(chunk)
                
                # Nouveau chunk avec chevauchement
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence_in_text
                current_start = text_position - len(overlap_text)
            else:
                if current_chunk:
                    current_chunk += " " + sentence_in_text
                else:
                    current_chunk = sentence_in_text
                    current_start = text_position
            
            text_position += len(sentence_in_text)
        
        # Dernier chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                current_start,
                len(text),
                len(chunks) + 1
            )
            chunks.append(chunk)
        
        return chunks
    
    def _detect_heading(self, paragraph: str) -> Tuple[bool, Optional[str]]:
        """Détecte si un paragraphe est un titre."""
        lines = paragraph.strip().split('\n')
        first_line = lines[0].strip()
        
        for pattern in self.heading_patterns:
            if re.match(pattern, first_line, re.MULTILINE):
                return True, first_line
        
        # Détection heuristique
        if (len(first_line) < 100 and 
            len(first_line.split()) < 10 and
            first_line[0].isupper()):
            return True, first_line
        
        return False, None
    
    def _get_overlap_text(self, chunk: str) -> str:
        """Extrait le texte de chevauchement à la fin d'un chunk."""
        if len(chunk) <= self.overlap_size:
            return ""
        
        # Prendre les dernières phrases jusqu'à overlap_size
        sentences = re.split(self.sentence_endings, chunk[-self.overlap_size:])
        
        if len(sentences) > 1:
            # Garder les phrases complètes
            return ". ".join(sentences[1:]) + ". "
        else:
            # Fallback: prendre les derniers mots
            words = chunk.split()
            overlap_words = []
            char_count = 0
            
            for word in reversed(words):
                if char_count + len(word) > self.overlap_size:
                    break
                overlap_words.insert(0, word)
                char_count += len(word) + 1
            
            return " ".join(overlap_words) + " " if overlap_words else ""
    
    def _create_chunk(self, content: str, start: int, end: int, chunk_id: int) -> TextChunk:
        """Crée un objet TextChunk avec métadonnées."""
        is_heading, heading_text = self._detect_heading(content)
        
        return TextChunk(
            content=content,
            start_index=start,
            end_index=end,
            chunk_id=chunk_id,
            total_chunks=0,  # Sera mis à jour dans post_process
            word_count=len(content.split()),
            has_heading=is_heading,
            heading_text=heading_text
        )
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Post-traitement des chunks."""
        total_chunks = len(chunks)
        
        # Mettre à jour le nombre total de chunks
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
        # Fusionner les chunks trop petits
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Si le chunk est trop petit et qu'il y a un chunk suivant
            if (len(current_chunk.content) < self.min_chunk_size and 
                i + 1 < len(chunks) and
                len(current_chunk.content) + len(chunks[i + 1].content) <= self.max_chunk_size):
                
                # Fusionner avec le chunk suivant
                next_chunk = chunks[i + 1]
                merged_content = current_chunk.content + "\n\n" + next_chunk.content
                
                merged_chunk = TextChunk(
                    content=merged_content,
                    start_index=current_chunk.start_index,
                    end_index=next_chunk.end_index,
                    chunk_id=len(merged_chunks) + 1,
                    total_chunks=0,  # Sera mis à jour à la fin
                    word_count=len(merged_content.split()),
                    has_heading=current_chunk.has_heading or next_chunk.has_heading,
                    heading_text=current_chunk.heading_text or next_chunk.heading_text
                )
                
                merged_chunks.append(merged_chunk)
                i += 2  # Passer les deux chunks fusionnés
            else:
                # Garder le chunk tel quel
                current_chunk.chunk_id = len(merged_chunks) + 1
                merged_chunks.append(current_chunk)
                i += 1
        
        # Mettre à jour le nombre total final
        for chunk in merged_chunks:
            chunk.total_chunks = len(merged_chunks)
        
        return merged_chunks
    
    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, any]:
        """Calcule les statistiques de découpage."""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        word_counts = [chunk.word_count for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "total_words": sum(word_counts),
            "average_chunk_size": sum(chunk_sizes) // len(chunks),
            "average_words_per_chunk": sum(word_counts) // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunks_with_headings": sum(1 for chunk in chunks if chunk.has_heading)
        }


class ChunkingManager:
    """
    Gestionnaire de chunking avec différentes stratégies.
    """
    
    def __init__(self):
        self.logger = setup_logger("chunking_manager")
        
        # Chunkers spécialisés
        self.chunkers = {
            "default": TextChunker(max_chunk_size=4000, overlap_size=200),
            "small": TextChunker(max_chunk_size=2000, overlap_size=100),
            "large": TextChunker(max_chunk_size=20000, overlap_size=300),
            "precise": TextChunker(max_chunk_size=3000, overlap_size=150, min_chunk_size=800)
        }
    
    def chunk_document(
        self, 
        content: str, 
        strategy: str = "default", 
        preserve_structure: bool = True
    ) -> List[TextChunk]:
        """
        Découpe un document selon la stratégie spécifiée.
        
        Args:
            content: Contenu à découper
            strategy: Stratégie de découpage (default, small, large, precise)
            preserve_structure: Préserver la structure du document
            
        Returns:
            Liste des chunks créés
        """
        if strategy not in self.chunkers:
            self.logger.warning(f"Stratégie inconnue '{strategy}', utilisation de 'default'")
            strategy = "default"
        
        chunker = self.chunkers[strategy]
        chunks = chunker.chunk_text(content, preserve_structure)
        
        # Statistiques
        stats = chunker.get_chunking_stats(chunks)
        self.logger.info(f"Chunking '{strategy}': {stats['total_chunks']} chunks créés")
        
        return chunks
    
    def auto_select_strategy(self, content: str) -> str:
        """Sélectionne automatiquement la meilleure stratégie de chunking."""
        content_length = len(content)
        word_count = len(content.split())
        
        # Heuristiques pour sélectionner la stratégie
        if content_length < 5000:
            return "small"
        elif content_length > 20000:
            return "large"
        elif word_count > 3000:  # Texte dense
            return "precise"
        else:
            return "default"