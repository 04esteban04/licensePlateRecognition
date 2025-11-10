document.addEventListener("DOMContentLoaded", () => {
	
	// Elements
	const messageBox = document.getElementById("message");

	const defaultWrapper = document.getElementById("processDefaultBtnWrapper");
	const processDefaultBtn = document.getElementById("processDefaultDatasetBtn");

	const resultCardContainer = document.getElementById("resultCardContainer");
	const resultCardGrid = document.getElementById("resultCardGrid");

	const results = document.getElementById("results");
	const toggleViewBtn = document.getElementById("toggleViewBtn");
	const resultTableContainer = document.getElementById("resultTableContainer");
	const resultTableBody = document.getElementById("resultTableBody");
		
	const btnIndividualContainer = document.getElementById("Individual");
	const btnDefaultContainer = document.getElementById("Default");
	
	const btnIndividual = document.getElementById("button-individual");
	const btnDefault = document.getElementById("button-default");

	const forms = document.querySelectorAll("form");
	const fileForm = document.getElementById("fileUploadFormIndividual");

	const fileUploadFormIndividual = document.querySelector('#fileUploadFormIndividual input[type="file"]');
	const uploadFileBtn = document.getElementById('uploadFileBtn');
	const fileNameSpan = document.getElementById('fileName');

	const getStartedBtn = document.getElementById("getStartedBtn");
	const learnMoreBtn = document.getElementById("learnMoreBtn");

	let currentView = "card"; 

	function toggleView() {
		if (currentView === "card") {
			resultCardContainer.classList.add("d-none");
			resultTableContainer.classList.remove("d-none");
			toggleViewBtn.textContent = "Toggle card view";
			currentView = "table";
		} else {
			resultCardContainer.classList.remove("d-none");
			resultTableContainer.classList.add("d-none");
			toggleViewBtn.textContent = "Toggle table view";
			currentView = "card";
		}
	}
	
	// Show processing confirmation message
	function showMessage(msg, isError = false) {
		messageBox.textContent = msg;
		messageBox.classList.remove("d-none", "alert-info", "alert-danger");
		messageBox.classList.add(isError ? "alert-danger" : "alert-info");
	}

	// Reset all modes and hide forms
	function resetMode() {
		fileForm.classList.add("d-none");
		defaultWrapper.classList.add("d-none");
		messageBox.classList.add("d-none");
		
		results.classList.add("d-none");
	
		// Clear previous values
		resultCardGrid.innerHTML = "";
		resultTableBody.innerHTML = "";
		currentView = "table";
		toggleView();

		btnIndividual.classList.remove("btn-light");
		btnDefault.classList.remove("btn-light");
		
		btnIndividual.classList.add("btn-outline-light");
		btnDefault.classList.add("btn-outline-light");
	}

	// Reset results display
	function resetResults() {
		resultCardGrid.innerHTML = "";
		resultTableBody.innerHTML = "";
		results.classList.add("d-none");

		currentView = "table"
		toggleView();
	}

	// Change loading spinner in buttons when submitting
	function loadingButtons(){

		// Prevent multiple form submits
		forms.forEach(form => {
			form.addEventListener("submit", function (e) {
				const button = form.querySelector(".submit-btn");
				if (button) {
					const textSpan = button.querySelector(".btn-text");
					const spinner = button.querySelector(".spinner-border");
					textSpan.classList.add("d-none");
					spinner.classList.remove("d-none");
					button.disabled = true;
				}
			});
		});

		// Default dataset button
		if (processDefaultBtn) {
			processDefaultBtn.addEventListener("click", function () {
				const textSpan = processDefaultBtn.querySelector(".btn-text");
				const spinner = processDefaultBtn.querySelector(".spinner-border");

				if (textSpan && spinner) {
					textSpan.classList.add("d-none");
					spinner.classList.remove("d-none");
					processDefaultBtn.disabled = true;
				}
			});
		}
	}

	// Reset all buttons with class "submit-btn"
	function resetLoadingButtons() {
		const buttons = document.querySelectorAll(".submit-btn, #processDefaultDatasetBtn");
		buttons.forEach(button => {
			const textSpan = button.querySelector(".btn-text");
			const spinner = button.querySelector(".spinner-border");

			if (textSpan && spinner) {
				textSpan.classList.remove("d-none");
				spinner.classList.add("d-none");
				button.disabled = false;
			}
		});
	}

	// Update button status when data is received
	function showSuccessCheck(button) {
		const textSpan = button.querySelector(".btn-text");
		const spinner = button.querySelector(".spinner-border");
		const checkIcon = button.querySelector(".check-icon");

		if (textSpan && spinner && checkIcon) {
			spinner.classList.add("d-none");
			checkIcon.classList.remove("d-none");
		}

		setTimeout(() => {
			checkIcon.classList.add("d-none");
			button.disabled = false;

			results.scrollIntoView({ behavior: "smooth" });
		}, 1000);
	}

	// Process individual file upload
	function handleFileFormSubmit(formElement) {
		formElement.addEventListener("submit", async (e) => {
			try {
				e.preventDefault();

				messageBox.classList.add("d-none");
				resetResults();
				
				const fileInput = formElement.querySelector("input[type='file']");
				const file = fileInput?.files[0];

				if (!file) {
					showMessage("Please select a file to upload.", true);
					return;
				}
								
				const formData = new FormData();
				formData.append("file", file);

				const response = await fetch("/process/image", {
					method: "POST",
					body: formData,
				});

				const result = await response.json();

				if (response.ok) {
			
					showMessage(result.message || "File processed successfully.");

					// === Check if images exist ===
					if (result.images) {
						results.classList.remove("d-none");

						// === CARD VIEW ===
						const col = document.createElement("div");
						col.classList.add("col");

						col.innerHTML = `
							<div class="card about-card h-100 shadow-sm">
								<div class="card-body text-center">
									<h5 class="card-title mb-3">Predicted Plate: ${result.predictedPlateNumber}</h5>
									<hr>
									<div class="d-flex flex-column align-items-center">
										<h6 class="text-white mb-2">Detected Plate</h6>
										<img src="${result.images.detectedPlate}" 
											alt="Detected Plate" 
											class="card-img-top rounded mb-2" 
											style="max-width: 300px; object-fit: contain;"
										>
										
										<h6 class="text-white mb-2">Cropped Plate</h6>
										<img src="${result.images.croppedPlate}" 
											alt="Cropped Plate" 
											class="card-img-top rounded mb-2" 
											style="max-width: 300px; object-fit: contain;"
										>

										<h6 class="text-white mb-2">Segmentation with Boxes</h6>
										<img src="${result.images.segmentationBoxes}" 
											alt="Segmentation with Boxes" 
											class="card-img-top rounded mb-2" 
											style="max-width: 300px; object-fit: contain;"
										>

										<h6 class="text-white mb-2">Binary Plate </h6>
										<img src="${result.images.segmentationThreshold}" 
											alt="Segmentation Threshold" 
											class="card-img-top rounded mb-2" 
											style="max-width: 300px; object-fit: contain;"
										>

									</div>

									${
										result.images.charInference?.length
											? `
												<hr>
												<h6>Detected Characters:</h6>
												<div class="d-flex flex-wrap justify-content-center gap-2 mt-2">
													${result.images.charInference
														.map(url => `<img src="${url}" class="border rounded" 
																		style="width:60px;height:60px;object-fit:contain;">`)
														.join("")}
												</div>
											`
											: ""
									}
								</div>
							</div>
						`;

						resultCardGrid.appendChild(col);

						// === TABLE VIEW ===
						const row = document.createElement("tr");
						row.innerHTML = `
							<td><img src="${result.images.detectedPlate}" alt="Detected Plate" class="img-thumbnail" style="max-width: 100px;"></td>
							<td><img src="${result.images.croppedPlate}" alt="Cropped Plate" class="img-thumbnail" style="max-width: 100px;"></td>
							<td><img src="${result.images.segmentationThreshold}" alt="Segmentation Threshold" class="img-thumbnail" style="max-width: 100px;"></td>
							<td><img src="${result.images.segmentationBoxes}" alt="Segmentation Boxes" class="img-thumbnail" style="max-width: 100px;"></td>
							<td>
								<div class="d-flex flex-wrap justify-content-center gap-2 mt-2">
									${result.images.charInference
										.map(url => `<img src="${url}" class="border rounded" 
														style="width:60px;height:60px;object-fit:contain;">`)
										.join("")}
								</div>
							</td>
							<td>${result.predictedPlateNumber}</td>
							<td>${result.meanPrecision}</td>
						`;
						resultTableBody.appendChild(row);

						resultCardContainer.classList.remove("d-none");

						const button = formElement.querySelector(".submit-btn");
						if (button) showSuccessCheck(button);
						
					} else {
						showMessage("No images returned from the server.", true);
					} 

				} else {
					showMessage(result.error || "An error occurred during processing.", true);
				}
			} 
			
			catch (error) {
				showMessage("Failed to process file: " + error.message, true);
				resetResults();
			} 

			finally {
				resetLoadingButtons(); 
			}
		});
	}

	// Tab navigation from main page buttons
	getStartedBtn.addEventListener("click", async (e) => {
		e.preventDefault();

		document.getElementById("tab-analysis").classList.add("active");
		document.getElementById("classify").classList.add("show", "active");

		document.getElementById("tab-home").classList.remove("active");
		document.getElementById("home").classList.remove("show", "active");
	});

	learnMoreBtn.addEventListener("click", (e) => {
  		e.preventDefault();

  		document.getElementById("tab-about").classList.add("active");
		document.getElementById("about").classList.add("show", "active");

		document.getElementById("tab-home").classList.remove("active");
		document.getElementById("home").classList.remove("show", "active");
	});

	 // Button event listeners
	btnIndividual.addEventListener("click", () => {
		resetMode();
		fileForm.classList.remove("d-none");
		btnIndividualContainer.classList.remove("gray-out");
		btnIndividual.classList.remove("btn-outline-light");
		btnIndividual.classList.add("btn-light");
		btnDefaultContainer.classList.add("gray-out");
	});

	btnDefault.addEventListener("click", () => {
		resetMode();
		defaultWrapper.classList.remove("d-none");
		btnDefaultContainer.classList.remove("gray-out");
		btnDefault.classList.remove("btn-outline-light");
		btnDefault.classList.add("btn-light");
		btnIndividualContainer.classList.add("gray-out");
	});

	// Process default dataset
	processDefaultBtn.addEventListener("click", async () => {
		try {
			const response = await fetch("/process/default", {
				method: "POST",
			});

			const result = await response.json();
			
			if (response.ok) {
				
				showMessage(result.message || "Default image processed successfully.");

				// === Clear previous results ===
				resultCardGrid.innerHTML = "";
				resultTableBody.innerHTML = "";

				// === Check if images exist ===
				if (result.images) {
					results.classList.remove("d-none");

					// === CARD VIEW ===
					const col = document.createElement("div");
					col.classList.add("col");

					col.innerHTML = `
						<div class="card about-card h-100 shadow-sm">
							<div class="card-body text-center">
								<h5 class="card-title mb-3">Predicted Plate: ${result.predictedPlateNumber}</h5>
								<hr>
								<div class="d-flex flex-column align-items-center">
									<h6 class="text-white mb-2">Detected Plate</h6>
									<img src="${result.images.detectedPlate}" 
										alt="Detected Plate" 
										class="card-img-top rounded mb-2" 
										style="max-width: 300px; object-fit: contain;"
									>
									
									<h6 class="text-white mb-2">Cropped Plate</h6>
									<img src="${result.images.croppedPlate}" 
										alt="Cropped Plate" 
										class="card-img-top rounded mb-2" 
										style="max-width: 300px; object-fit: contain;"
									>

									<h6 class="text-white mb-2">Segmentation with Boxes</h6>
									<img src="${result.images.segmentationBoxes}" 
										alt="Segmentation with Boxes" 
										class="card-img-top rounded mb-2" 
										style="max-width: 300px; object-fit: contain;"
									>

									<h6 class="text-white mb-2">Segmentation Threshold</h6>
									<img src="${result.images.segmentationThreshold}" 
										alt="Segmentation Threshold" 
										class="card-img-top rounded mb-2" 
										style="max-width: 300px; object-fit: contain;"
									>

								</div>

								${
									result.images.charInference?.length
										? `
											<hr>
											<h6>Detected Characters:</h6>
											<div class="d-flex flex-wrap justify-content-center gap-2 mt-2">
												${result.images.charInference
													.map(url => `<img src="${url}" class="border rounded" 
																	style="width:60px;height:60px;object-fit:contain;">`)
													.join("")}
											</div>
										`
										: ""
								}
							</div>
						</div>
					`;

					resultCardGrid.appendChild(col);

					// === TABLE VIEW ===
					const row = document.createElement("tr");
					row.innerHTML = `
							<td><img src="${result.images.detectedPlate}" alt="Detected Plate" class="img-thumbnail" style="max-width: 100px;"></td>
							<td><img src="${result.images.croppedPlate}" alt="Cropped Plate" class="img-thumbnail" style="max-width: 100px;"></td>
							<td><img src="${result.images.segmentationThreshold}" alt="Segmentation Threshold" class="img-thumbnail" style="max-width: 100px;"></td>
							<td><img src="${result.images.segmentationBoxes}" alt="Segmentation Boxes" class="img-thumbnail" style="max-width: 100px;"></td>
							<td>
								<div class="d-flex flex-wrap justify-content-center gap-2 mt-2">
									${result.images.charInference
										.map(url => `<img src="${url}" class="border rounded" 
														style="width:60px;height:60px;object-fit:contain;">`)
										.join("")}
								</div>
							</td>
							<td>${result.predictedPlateNumber}</td>
							<td>${result.meanPrecision}</td>
					`;
					resultTableBody.appendChild(row);

					resultCardContainer.classList.remove("d-none");

					showSuccessCheck(processDefaultBtn);

				} else {
					showMessage("No images returned from the server.", true);
					resetResults();
				}
 
			} else {
				showMessage(result.error || "An error occurred.", true);
				resetResults();
			}
		} catch (error) {
			showMessage("Failed to process default image: " + error.message, true);
			resetResults();
		}
		finally {
			resetLoadingButtons(); 
		}
	});

	// Update file name display when a file is selected
	fileUploadFormIndividual.addEventListener('change', () => {
		if (fileUploadFormIndividual.files.length > 0) {
			fileNameSpan.textContent = fileUploadFormIndividual.files[0].name;
		} else {
			fileNameSpan.textContent = 'No file chosen';
		}
	});

	// Toggle between card and table views
	toggleViewBtn.addEventListener("click", toggleView);
	
	// Handle file upload button clicks
	uploadFileBtn.addEventListener('click', () => fileUploadFormIndividual.click());

	// Handle file form submissions by mode
	handleFileFormSubmit(fileForm, false);

	// Change loading spinner in buttons when submitting
	loadingButtons();

});
