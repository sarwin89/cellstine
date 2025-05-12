const canvas = document.getElementById('hexGridCanvas');
const ctx = canvas.getContext('2d');
const rotationSlider = document.getElementById('rotationSlider');
const angleInput = document.getElementById('angleInput');

// Adjust canvas size to fill the entire screen
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Hexagonal grid settings
const hexSize = 10;
const hexWidth = Math.sqrt(3) * hexSize;  // Horizontal spacing
const hexHeight = 2 * hexSize;  // Vertical spacing

// Adjust the spacing and number of rows/cols based on the canvas size
const cols = Math.floor(canvas.width / hexWidth);
const rows = Math.floor(canvas.height / (hexHeight * 0.75));

// Function to calculate the center of the closest hexagon
function getHexagonCenter(row, col) {
    const x = col * hexWidth + (row % 2) * (hexWidth / 2);
    const y = row * hexHeight * 0.75;
    return { x: x + hexWidth / 2, y: y + hexHeight / 2 };
}

function drawHexagon(x, y, size) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
        const angle = Math.PI / 3 * i + Math.PI / 6; // Offset for correct orientation
        const dx = x + size * Math.cos(angle);
        const dy = y + size * Math.sin(angle);
        ctx.lineTo(dx, dy);
    }
    ctx.closePath();
    ctx.lineWidth = 2;  // Thicker lines for the hexagons
    ctx.strokeStyle = "#AAA";  // White color for hexagons
    ctx.stroke();
}

/*function drawSupercell() {
    // Smallest supercell vectors (basis for the hexagonal grid)
    const a1 = { x: hexWidth, y: 0 };
    const a2 = { x: hexWidth / 2, y: hexHeight * 0.75 };

    // Draw the supercell vectors in green
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, canvas.height / 2);
    ctx.lineTo(canvas.width / 2 + a1.x, canvas.height / 2 + a1.y);
    ctx.strokeStyle = "green";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, canvas.height / 2);
    ctx.lineTo(canvas.width / 2 + a2.x, canvas.height / 2 + a2.y);
    ctx.strokeStyle = "green";
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Display the supercell area
    const area = hexSize * hexSize * Math.sqrt(3) / 2;  // Area of a single hexagon
    ctx.fillStyle = "green";
    ctx.font = "20px sans-serif";
    ctx.fillText(`Supercell Area: ${area.toFixed(2)}`, canvas.width / 2 + a1.x + 10, canvas.height / 2 + a1.y + 20);
}*/

function drawLattice(rotationAngle) {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas

    // Find the nearest hexagon to the center of the canvas
    const centerHex = getHexagonCenter(Math.floor(rows / 2), Math.floor(cols / 2));

    // Draw the fixed lattice (no rotation)
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const x = col * hexWidth;
            const y = row * hexHeight * 0.75;

            // Apply staggering for the hexagonal lattice
            const xOffset = (row % 2) * hexWidth / 2;
            drawHexagon(x + xOffset, y, hexSize);
        }
    }

    // Draw the rotated lattice centered on the selected hexagon
    ctx.save();
    ctx.translate(centerHex.x, centerHex.y + hexHeight * 0.25); // Move origin to the center of the nearest hexagon
    ctx.rotate(rotationAngle); // Rotate the canvas
    ctx.translate(-centerHex.x, -centerHex.y - hexHeight * 0.25); // Move origin back

    for (let row = -2*rows; row < 2*rows; row++) {
        for (let col = -2*cols; col < 2*cols; col++) {
            const x = col * hexWidth;
            const y = row * hexHeight * 0.75;

            // Apply staggering for the hexagonal lattice
            const xOffset = (row % 2) * hexWidth / 2;
            drawHexagon(x + xOffset, y, hexSize);
        }
    }

    ctx.restore();

    // Draw and display the supercell
    //drawSupercell();
}

// Initial drawing with no rotation
drawLattice(0);

// Update rotation angle when the slider is changed
rotationSlider.addEventListener('input', (e) => {
    const angle = (e.target.value * Math.PI) / 180; // Convert to radians
    angleInput.value = e.target.value; // Update input box
    drawLattice(angle);
});

// Update rotation when typing in the input box
angleInput.addEventListener('input', (e) => {
    const angle = parseFloat(e.target.value);
    rotationSlider.value = angle;
    drawLattice((angle * Math.PI) / 180); // Convert to radians
});