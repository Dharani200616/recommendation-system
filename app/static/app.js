// ==========================================================================
// CINEMATCH AI - INTERACTIVE SANDBOX FRONTEND ORCHESTRATOR
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    // 1. DOM Element Cache
    const selectUserId = document.getElementById('select-user-id');
    const selectAlgorithm = document.getElementById('select-algorithm');
    const selectGenre = document.getElementById('select-genre');
    const btnFetchRecs = document.getElementById('btn-fetch-recs');
    const recsLoader = document.getElementById('recs-loader');
    const recsOutputContainer = document.getElementById('recs-output-container');
    const recsCardsContainer = document.getElementById('recs-cards-container');
    const userHistoryList = document.getElementById('user-history-list');

    // Simulator Elements
    const simUserId = document.getElementById('sim-user-id');
    const simMovieSearch = document.getElementById('sim-movie-search');
    const simSearchDropdown = document.getElementById('sim-search-dropdown-results');
    const selectedMovieDetails = document.getElementById('selected-movie-details');
    const selectedMovieTitle = document.getElementById('selected-movie-title');
    const selectedMovieGenres = document.getElementById('selected-movie-genres');
    const selectedMovieId = document.getElementById('selected-movie-id');
    const starBtns = document.querySelectorAll('.star-btn');
    const selectedStarValue = document.getElementById('selected-star-value');
    const btnSubmitRating = document.getElementById('btn-submit-rating');
    const simulatorTerminalBody = document.getElementById('simulator-terminal-body');

    // Toast Notification
    const toastNotify = document.getElementById('toast-notify');
    const toastMessage = document.getElementById('toast-message');

    // Catalog Header Search
    const catalogSearchInput = document.getElementById('movie-catalog-search');
    const catalogSearchDropdown = document.getElementById('search-dropdown-results');

    // 2. Load Recommendations & History AJAX
    async function loadRecommendations() {
        const userId = selectUserId.value;
        const method = selectAlgorithm.value;
        const genre = selectGenre.value;

        // Toggle state
        recsLoader.style.display = 'flex';
        recsOutputContainer.style.opacity = '0.3';
        
        try {
            const response = await fetch(`/recommendations?user_id=${userId}&method=${method}&genre=${genre}`);
            const data = await response.json();
            
            renderRecommendations(data.recommendations);
            renderUserHistory(data.history);
        } catch (error) {
            console.error("Failed to load recommendations:", error);
            logTerminal("warn-log", `Error: Failed to fetch recommendations for User ${userId}`);
        } finally {
            recsLoader.style.display = 'none';
            recsOutputContainer.style.opacity = '1';
        }
    }

    function renderRecommendations(recs) {
        recsCardsContainer.innerHTML = '';
        
        if (!recs || recs.length === 0) {
            recsCardsContainer.innerHTML = `
                <div class="empty-recs-message" style="grid-column: span 2; text-align: center; padding: 3rem; color: var(--text-muted);">
                    <i class="fa-solid fa-face-meh" style="font-size: 2.5rem; margin-bottom: 1rem; color: var(--primary-neon);"></i>
                    <p>No unrated matches found for this filter combination.</p>
                </div>
            `;
            return;
        }

        recs.forEach((rec, idx) => {
            const card = document.createElement('div');
            card.className = 'movie-card animated-slideUp';
            card.style.animationDelay = `${idx * 0.08}s`;
            
            const starScore = Math.round(rec.predicted_rating);
            const starsHTML = '★'.repeat(starScore) + '☆'.repeat(5 - starScore);
            
            card.innerHTML = `
                <div class="card-header-meta">
                    <span class="card-genres">${rec.genres.replace(/\|/g, ' • ')}</span>
                    <h4>${rec.title} ${rec.release_year ? `(${rec.release_year})` : ''}</h4>
                </div>
                <div class="card-score-info">
                    <span class="score-number">${rec.predicted_rating.toFixed(2)}</span>
                    <span class="score-stars">${starsHTML}</span>
                </div>
            `;
            
            recsCardsContainer.appendChild(card);
        });
    }

    function renderUserHistory(history) {
        userHistoryList.innerHTML = '';
        
        if (!history || history.length === 0) {
            userHistoryList.innerHTML = `
                <p style="color: var(--text-muted); font-size: 0.8rem; font-style: italic;">No training rating logs found for this user.</p>
            `;
            return;
        }

        history.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            
            const ratingScore = Math.round(item.rating);
            const starsHTML = '★'.repeat(ratingScore) + '☆'.repeat(5 - ratingScore);
            
            div.innerHTML = `
                <div class="history-item-meta">
                    <strong>${item.title}</strong>
                    <span class="movie-genres-pill">${item.genres.replace(/\|/g, ' • ')}</span>
                </div>
                <div class="history-rating">${starsHTML}</div>
            `;
            userHistoryList.appendChild(div);
        });
    }

    // 3. Simulator Autocomplete Movie Search
    async function searchMovies(query, dropdownEl, selectCallback) {
        if (!query) {
            dropdownEl.style.display = 'none';
            return;
        }
        
        try {
            const response = await fetch(`/movies/search?q=${encodeURIComponent(query)}`);
            const movies = await response.json();
            
            dropdownEl.innerHTML = '';
            if (movies.length === 0) {
                dropdownEl.style.display = 'none';
                return;
            }
            
            movies.forEach(movie => {
                const item = document.createElement('div');
                item.className = 'search-dropdown-item';
                item.innerText = `${movie.title} (${movie.genres.replace(/\|/g, ' • ')})`;
                item.addEventListener('click', () => selectCallback(movie));
                dropdownEl.appendChild(item);
            });
            
            dropdownEl.style.display = 'block';
        } catch (error) {
            console.error("Movie search failed:", error);
        }
    }

    // Bind Auto-complete callbacks
    simMovieSearch.addEventListener('input', (e) => {
        searchMovies(e.target.value, simSearchDropdown, (movie) => {
            selectedMovieTitle.innerText = movie.title;
            selectedMovieGenres.innerText = movie.genres.replace(/\|/g, ' • ');
            selectedMovieId.value = movie.movie_id;
            selectedMovieDetails.style.display = 'block';
            
            simMovieSearch.value = '';
            simSearchDropdown.style.display = 'none';
            checkFormValidity();
        });
    });

    // Global Top Header Autocomplete Search
    catalogSearchInput.addEventListener('input', (e) => {
        searchMovies(e.target.value, catalogSearchDropdown, (movie) => {
            // Fill rating simulator and scroll directly to it
            selectedMovieTitle.innerText = movie.title;
            selectedMovieGenres.innerText = movie.genres.replace(/\|/g, ' • ');
            selectedMovieId.value = movie.movie_id;
            selectedMovieDetails.style.display = 'block';
            
            catalogSearchInput.value = '';
            catalogSearchDropdown.style.display = 'none';
            checkFormValidity();
            
            // Highlight Simulator
            document.getElementById('rating-section').scrollIntoView({ behavior: 'smooth' });
        });
    });

    // Close dropdowns on click outside
    document.addEventListener('click', (e) => {
        if (e.target !== simMovieSearch) simSearchDropdown.style.display = 'none';
        if (e.target !== catalogSearchInput) catalogSearchDropdown.style.display = 'none';
    });

    // 4. Rating Simulator Widget Interactive Stars
    starBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const val = parseInt(btn.getAttribute('data-value'));
            selectedStarValue.value = val;
            
            // Highlight active stars
            starBtns.forEach(star => {
                const starVal = parseInt(star.getAttribute('data-value'));
                if (starVal <= val) {
                    star.classList.add('active');
                    star.classList.replace('fa-regular', 'fa-solid');
                } else {
                    star.classList.remove('active');
                    star.classList.replace('fa-solid', 'fa-regular');
                }
            });
            
            checkFormValidity();
        });
    });

    function checkFormValidity() {
        const movieId = selectedMovieId.value;
        const stars = parseInt(selectedStarValue.value);
        btnSubmitRating.disabled = !(movieId && stars > 0);
    }

    // 5. Submit Simulated Rating & Live Retraining Simulation logs
    btnSubmitRating.addEventListener('click', async () => {
        const userId = parseInt(simUserId.value);
        const movieId = parseInt(selectedMovieId.value);
        const rating = parseFloat(selectedStarValue.value);
        const movieTitle = selectedMovieTitle.innerText;

        btnSubmitRating.disabled = true;
        
        // Terminal log stream
        logTerminal("info-log", `> Dispatching simulated rating to backend...`);
        logTerminal("info-log", `> User: ${userId} rated "${movieTitle}" -> ${rating} Stars.`);
        
        try {
            const response = await fetch('/rate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userId, movie_id: movieId, rating: rating })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Simulate ML retraining sequence
                logTerminal("info-log", `> Retraining User-Similarity Matrices...`);
                await delay(300);
                logTerminal("info-log", `> Centering ratings index vector...`);
                await delay(200);
                logTerminal("info-log", `> Re-calculating Mean Pearson Similarity scores...`);
                await delay(400);
                logTerminal("success-log", `> Model re-fit completed in 920ms! Global weights aggregated.`);
                
                showToast(`Simulation Success! Rated "${movieTitle}" ${rating}★`);
                
                // Clear selection
                resetSimulatorForm();
                
                // If the target sandbox user is the same as simulated user, reload recommendations instantly
                if (parseInt(selectUserId.value) === userId) {
                    loadRecommendations();
                }
            } else {
                logTerminal("warn-log", `> Engine error: ${data.message}`);
            }
        } catch (error) {
            logTerminal("warn-log", `> Engine crash during backpropagation.`);
        }
    });

    function resetSimulatorForm() {
        selectedMovieId.value = '';
        selectedStarValue.value = '0';
        selectedMovieDetails.style.display = 'none';
        
        starBtns.forEach(star => {
            star.classList.remove('active');
            star.classList.replace('fa-solid', 'fa-regular');
        });
        
        btnSubmitRating.disabled = true;
    }

    // Helper functions
    function logTerminal(typeClass, message) {
        const p = document.createElement('p');
        p.className = `t-log ${typeClass}`;
        p.innerText = message;
        simulatorTerminalBody.appendChild(p);
        simulatorTerminalBody.scrollTop = simulatorTerminalBody.scrollHeight;
    }

    function showToast(msg) {
        toastMessage.innerText = msg;
        toastNotify.style.display = 'flex';
        setTimeout(() => {
            toastNotify.style.display = 'none';
        }, 4000);
    }

    const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    // 6. Navigation items scrolling highlights
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
        });
    });

    // 7. Initialize
    btnFetchRecs.addEventListener('click', loadRecommendations);
    loadRecommendations();
});
