
// ==========================================
// RESOLVEAI - COMPLAINT INTELLIGENCE
// ==========================================

// DOM ELEMENTS

const complaintInput = document.getElementById("complaintText");

const charCount = document.getElementById("charCount");
const errorBox = document.getElementById("error");

const predictionResults = document.getElementById("predictionResults");
const similarResults = document.getElementById("similarResults");
const emptyState = document.getElementById("emptyState");
const reportState = document.getElementById("reportState");

const predictionGrid = document.getElementById("predictionGrid");
const similarList = document.getElementById("similarList");

const newComplaintBtn = document.getElementById("newComplaintBtn");

const buttons = {
    classify: document.getElementById("classifyBtn"),
    similar: document.getElementById("similarBtn"),
    analyze: document.getElementById("analyzeBtn")
};

let isProcessing = false;


// ==========================================
// CHARACTER COUNTER
// ==========================================

complaintInput.addEventListener("input", () => {

    charCount.textContent =
        `${complaintInput.value.length} / 5000`;

    // Clear old errors when user edits input
    clearError();

});


// ==========================================
// ERROR HANDLING
// ==========================================

function showError(message) {

    errorBox.textContent = message;
    errorBox.hidden = false;

}

function clearError() {

    errorBox.textContent = "";
    errorBox.hidden = true;

}


// ==========================================
// LOADING STATE
// ==========================================

function setLoading(loading, activeButton = null) {

    isProcessing = loading;

    // Disable all action buttons during a request
    Object.values(buttons).forEach(button => {

        button.disabled = loading;

    });

    newComplaintBtn.disabled = loading;

    if (loading && activeButton) {

        activeButton.dataset.originalHtml =
            activeButton.innerHTML;

        activeButton.innerHTML = `
            <span class="loading-spinner"></span>
            Processing...
        `;

    }

    else if (!loading) {

        Object.values(buttons).forEach(button => {

            if (button.dataset.originalHtml) {

                button.innerHTML =
                    button.dataset.originalHtml;

                delete button.dataset.originalHtml;

            }

        });

    }

}


// ==========================================
// API FUNCTION
// ==========================================

async function apiCall(url, extraBody, button) {

    const complaintText = complaintInput.value.trim();

    clearError();

    if (!complaintText) {

        showError("Please enter a complaint first.");

        complaintInput.focus();

        return null;

    }

    if (isProcessing) {
        return null;
    }

    setLoading(true, button);

    try {

        const response = await fetch(url, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                complaint_text: complaintText,

                ...extraBody

            })

        });


        // Handle invalid or non-JSON responses
        let data;

        try {

            data = await response.json();

        }

        catch {

            throw new Error(
                "Unable to read server response. Please try again."
            );

        }


        if (!response.ok) {

            throw new Error(
                data.error || "Something went wrong."
            );

        }

        return data;

    }

    catch (error) {

        showError(error.message);

        return null;

    }

    finally {

        setLoading(false);

    }

}


// ==========================================
// RENDER PREDICTION RESULTS
// ==========================================

function renderPrediction(data) {

    const fields = [

        ["category", "Category"],
        ["priority", "Priority"],
        ["sentiment", "Sentiment"],
        ["emotion", "Emotion"],
        ["department", "Department"]

    ];


    predictionGrid.replaceChildren();


    fields.forEach(([key, label]) => {

        const item = document.createElement("div");

        item.className = "pred-item";


        const labelElement = document.createElement("div");

        labelElement.className = "label";

        labelElement.textContent = label;


        const valueElement = document.createElement("span");

        valueElement.className = "value";

        valueElement.textContent =
            data?.[key] ?? "N/A";


        item.append(labelElement, valueElement);

        predictionGrid.appendChild(item);

    });


    predictionResults.hidden = false;

    emptyState.hidden = true;

    reportState.textContent = "PREDICTION COMPLETE";

}


// ==========================================
// RENDER SIMILAR COMPLAINTS
// ==========================================

function renderSimilar(items) {

    similarList.replaceChildren();


    if (!Array.isArray(items) || items.length === 0) {

        const message = document.createElement("p");

        message.className = "similar-text";

        message.textContent =
            "No similar complaints found.";

        similarList.appendChild(message);

    }

    else {

        items.forEach((item, index) => {

            const card = document.createElement("article");

            card.className = "similar-item";


            const complaint = document.createElement("p");

            complaint.className = "similar-text";

            complaint.textContent =
                `#${index + 1} ${item.complaint_text ?? "Complaint text unavailable"}`;


            const meta = document.createElement("div");

            meta.className = "similar-meta";


            function addMeta(value, className = "") {

                if (
                    value === undefined ||
                    value === null ||
                    value === ""
                ) {
                    return;
                }

                const tag = document.createElement("span");

                tag.textContent = value;

                if (className) {

                    tag.classList.add(className);

                }

                meta.appendChild(tag);

            }


            addMeta(item.category);

            addMeta(item.priority);

            addMeta(
                `Similarity: ${item.similarity_score ?? "N/A"}`,
                "score-badge"
            );


            card.append(complaint, meta);

            similarList.appendChild(card);

        });

    }


    similarResults.hidden = false;

    emptyState.hidden = true;

    reportState.textContent =
        "SIMILARITY SEARCH COMPLETE";

}


// ==========================================
// CLASSIFY BUTTON
// ==========================================

buttons.classify.addEventListener("click", async () => {

    const data = await apiCall(
        "/predict",
        {},
        buttons.classify
    );

    if (data) {

        // Hide old similarity results
        similarResults.hidden = true;

        renderPrediction(data);

        document.getElementById("results").scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }

});


// ==========================================
// FIND SIMILAR BUTTON
// ==========================================

buttons.similar.addEventListener("click", async () => {

    const data = await apiCall(
        "/similar",
        { top_n: 5 },
        buttons.similar
    );

    if (data) {

        // Hide old prediction results
        predictionResults.hidden = true;

        renderSimilar(data.similar_complaints);

        document.getElementById("results").scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }

});


// ==========================================
// FULL ANALYSIS BUTTON
// ==========================================

buttons.analyze.addEventListener("click", async () => {

    const data = await apiCall(
        "/analyze",
        { top_n: 5 },
        buttons.analyze
    );

    if (data) {

        if (data.prediction) {

            renderPrediction(data.prediction);

        }

        renderSimilar(data.similar_complaints);

        reportState.textContent = "FULL ANALYSIS COMPLETE";

        document.getElementById("results").scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }

});


// ==========================================
// NEW COMPLAINT BUTTON
// ==========================================

newComplaintBtn.addEventListener("click", () => {

    if (isProcessing) {
        return;
    }

    // Clear complaint text
    complaintInput.value = "";

    // Reset character counter
    charCount.textContent = "0 / 5000";

    // Hide previous results
    predictionResults.hidden = true;
    similarResults.hidden = true;

    // Remove old prediction cards
    predictionGrid.replaceChildren();

    // Remove old similar complaints
    similarList.replaceChildren();

    // Show empty state
    emptyState.hidden = false;

    // Reset report status
    reportState.textContent = "AWAITING INPUT";

    // Clear errors
    clearError();

    // Focus textarea for new complaint
    complaintInput.focus();

    // Return to analyzer section
    document.getElementById("analyzer").scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

});
