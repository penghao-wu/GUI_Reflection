// carousel.js

document.addEventListener("DOMContentLoaded", () => {
    // 1) Select only our custom carousel containers
    const carousels = document.querySelectorAll(".my-carousel");
  
    carousels.forEach((carousel) => {
      // Ensure the container can position children absolutely 
      carousel.style.position = "relative";
      carousel.style.overflow = "hidden";
  
      // 2) Gather all .my-item slides inside this carousel
      const slides = Array.from(carousel.querySelectorAll(".my-item"));
      const totalSlides = slides.length;
      if (totalSlides === 0) return; // nothing to do if no slides
  
      // 3) Hide all slides except the first one
      slides.forEach((slide, idx) => {
        slide.style.display = idx === 0 ? "block" : "none";
        slide.style.width = "100%";
        slide.style.height = "auto";
        // Make sure each slide is positioned statically
        slide.style.position = "relative";
      });
  
      let currentIndex = 0;
  
      // 4) Create the ◀️ “previous” button
      const prevBtn = document.createElement("button");
      prevBtn.innerHTML = "&#10094;"; // Unicode left arrow ‹
      styleArrowButton(prevBtn, "left");
      carousel.appendChild(prevBtn);
  
      // 5) Create the ▶️ “next” button
      const nextBtn = document.createElement("button");
      nextBtn.innerHTML = "&#10095;"; // Unicode right arrow ›
      styleArrowButton(nextBtn, "right");
      carousel.appendChild(nextBtn);
  
      // 6) Function to show a slide by newIndex
      function showSlide(newIndex) {
        if (newIndex < 0) {
          newIndex = totalSlides - 1;
        } else if (newIndex >= totalSlides) {
          newIndex = 0;
        }
        slides[currentIndex].style.display = "none";
        slides[newIndex].style.display = "block";
        currentIndex = newIndex;
      }
  
      // 7) Wire up the click events
      prevBtn.addEventListener("click", () => {
        showSlide(currentIndex - 1);
      });
      nextBtn.addEventListener("click", () => {
        showSlide(currentIndex + 1);
      });
  
      // Optional: auto‐advance every 5 seconds
      // setInterval(() => {
      //   showSlide(currentIndex + 1);
      // }, 5000);
    });
  });
  
  // 8) Inline styling helper for arrow buttons
  function styleArrowButton(button, side) {
    button.style.position = "absolute";
    button.style.top = "50%";
    button.style[side] = "10px";     // either 'left' or 'right'
    button.style.transform = "translateY(-50%)";
    button.style.zIndex = "10";
  
    // Make it a circle with semi-transparent white background
    button.style.width = "40px";
    button.style.height = "40px";
    button.style.borderRadius = "50%";
    button.style.border = "none";
    button.style.backgroundColor = "rgba(255, 255, 255, 0.7)";
    button.style.color = "#333";
    button.style.fontSize = "1.5rem";
    button.style.cursor = "pointer";
    button.style.display = "flex";
    button.style.alignItems = "center";
    button.style.justifyContent = "center";
    button.style.padding = "0";
  
    // Hover effect
    button.addEventListener("mouseenter", () => {
      button.style.backgroundColor = "rgba(255, 255, 255, 0.9)";
    });
    button.addEventListener("mouseleave", () => {
      button.style.backgroundColor = "rgba(255, 255, 255, 0.7)";
    });
  }
  