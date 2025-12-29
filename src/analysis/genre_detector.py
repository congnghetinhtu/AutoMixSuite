"""
Genre detection using audio features
"""

import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GenreDetector:
    """Detects musical genre from audio features"""
    
    @staticmethod
    def detect(y: np.ndarray, sr: int, tempo: float, energy: float, spectral_centroid: float) -> str:
        """
        Detect genre hint for tempo matching strategy
        
        Returns:
            Genre category: 'electronic', 'pop', 'rock', 'jazz', 'classical', 'hiphop',
                          'vietnamese_ballad', 'vietnamese_pop', 'cuba_bolero', 'future_funk',
                          'house', 'unknown'
        """
        try:
            # Analyze only first 90 seconds for performance
            max_samples = min(len(y), sr * 90)
            y_short = y[:max_samples]
            
            # Percussive vs harmonic balance
            y_harmonic, y_percussive = librosa.effects.hpss(y_short, margin=2.0)
            total_energy = np.sqrt(np.mean(y_short**2)) + 1e-10
            harmonic_ratio = np.sqrt(np.mean(y_harmonic**2)) / total_energy
            percussive_ratio = np.sqrt(np.mean(y_percussive**2)) / total_energy
            
            # Normalize ratios
            total_ratio = harmonic_ratio + percussive_ratio
            if total_ratio > 0:
                harmonic_ratio /= total_ratio
                percussive_ratio /= total_ratio
            
            # Spectral features
            hop_length = 1024
            contrast = librosa.feature.spectral_contrast(y=y_short, sr=sr, hop_length=hop_length)
            avg_contrast = np.mean(contrast)
            contrast_var = np.std(np.mean(contrast, axis=0))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y_short, hop_length=hop_length)[0]
            zcr_var = np.std(zcr)
            zcr_mean = np.mean(zcr)
            
            # Genre scoring
            genre_scores = {}
            
            # Electronic/EDM
            score = 0.0
            if 115 <= tempo <= 145:
                score += 0.3
                if 125 <= tempo <= 135:
                    score += 0.2
            if percussive_ratio > 0.55:
                score += 0.3
            if contrast_var < 3.0:
                score += 0.2
            if score > 0.4:
                genre_scores['electronic'] = score
            
            # Hip-Hop
            score = 0.0
            if 80 <= tempo <= 110:
                score += 0.3
            if percussive_ratio > 0.5:
                score += 0.3
            if spectral_centroid < 2500:
                score += 0.2
            if zcr_mean < 0.08:
                score += 0.2
            if score > 0.4:
                genre_scores['hiphop'] = score
            
            # Rock
            score = 0.0
            if 110 <= tempo <= 170:
                score += 0.25
            if 0.35 <= harmonic_ratio <= 0.65:
                score += 0.25
            if avg_contrast > 20:
                score += 0.25
            if energy > 0.15:
                score += 0.25
            if score > 0.5:
                genre_scores['rock'] = score
            
            # Pop
            score = 0.0
            if 95 <= tempo <= 135:
                score += 0.25
            if 0.4 <= harmonic_ratio <= 0.6:
                score += 0.25
            if energy > 0.1:
                score += 0.25
            if 2000 <= spectral_centroid <= 3500:
                score += 0.25
            if score > 0.5:
                genre_scores['pop'] = score
            
            # Jazz
            score = 0.0
            if harmonic_ratio > 0.55:
                score += 0.3
            if contrast_var > 4.0:
                score += 0.2
            if zcr_var > 0.025:
                score += 0.3
            if 80 <= tempo <= 200:
                score += 0.2
            if score > 0.5:
                genre_scores['jazz'] = score
            
            # Classical
            score = 0.0
            if harmonic_ratio > 0.7:
                score += 0.4
            if avg_contrast > 23:
                score += 0.3
            if contrast_var > 3.5:
                score += 0.2
            if spectral_centroid > 2000:
                score += 0.1
            if score > 0.6:
                genre_scores['classical'] = score
            
            # Vietnamese Ballad
            score = 0.0
            if 70 <= tempo <= 95:
                score += 0.3
            if harmonic_ratio > 0.6:
                score += 0.3
            if spectral_centroid < 2800:
                score += 0.2
            if 0.08 <= energy <= 0.15:
                score += 0.2
            if score > 0.5:
                genre_scores['vietnamese_ballad'] = score
            
            # Vietnamese Pop
            score = 0.0
            if 95 <= tempo <= 130:
                score += 0.25
            if 0.45 <= harmonic_ratio <= 0.55:
                score += 0.25
            if 2200 <= spectral_centroid <= 3200:
                score += 0.25
            if energy > 0.12:
                score += 0.25
            if score > 0.5:
                genre_scores['vietnamese_pop'] = score
            
            # Cuban Bolero
            score = 0.0
            if 60 <= tempo <= 90:
                score += 0.3
            if harmonic_ratio > 0.55:
                score += 0.3
            if 0.45 <= percussive_ratio <= 0.55:
                score += 0.2
            if spectral_centroid < 3000:
                score += 0.1
            if 0.08 <= energy <= 0.18:
                score += 0.1
            if score > 0.5:
                genre_scores['cuba_bolero'] = score
            
            # Future Funk (Young Franco style): Groovy, funky basslines, disco samples, moderate-high tempo
            score = 0.0
            if 105 <= tempo <= 128:  # Funky groove tempo
                score += 0.3
                if 110 <= tempo <= 120:  # Sweet spot for funk
                    score += 0.1
            if 0.4 <= harmonic_ratio <= 0.6:  # Balanced funky groove
                score += 0.3
            if percussive_ratio > 0.4:  # Strong rhythm section
                score += 0.2
            if 2500 <= spectral_centroid <= 3800:  # Bright, funky sound
                score += 0.1
            if energy > 0.13:  # Energetic and danceable
                score += 0.1
            if score > 0.5:
                genre_scores['future_funk'] = score
            
            # House (Future House, Future Bounce)
            score = 0.0
            if 118 <= tempo <= 132:  # House tempo range
                score += 0.35
                if 124 <= tempo <= 128:  # Classic house sweet spot
                    score += 0.15
            if percussive_ratio > 0.55:  # Strong 4/4 kick pattern
                score += 0.25
            if 2800 <= spectral_centroid <= 4200:  # Bright synth-heavy sound
                score += 0.15
            if energy > 0.18:  # High energy, club-ready
                score += 0.1
            if score > 0.5:
                genre_scores['house'] = score
            
            # Select best genre
            if genre_scores:
                sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
                top_genre, top_score = sorted_genres[0]
                
                if top_score >= 0.5:
                    if len(sorted_genres) > 1:
                        second_score = sorted_genres[1][1]
                        if top_score - second_score < 0.15:
                            logger.info(f"    Genre hint: {top_genre} (confidence: {top_score:.2f}, ambiguous with {sorted_genres[1][0]})")
                        else:
                            logger.info(f"    Genre hint: {top_genre} (confidence: {top_score:.2f})")
                    else:
                        logger.info(f"    Genre hint: {top_genre} (confidence: {top_score:.2f})")
                    return top_genre
                else:
                    logger.info(f"    Genre uncertain: best guess {top_genre} (confidence: {top_score:.2f})")
            
            return 'unknown'
            
        except Exception as e:
            logger.warning(f"    Genre detection failed: {e}")
            return 'unknown'
