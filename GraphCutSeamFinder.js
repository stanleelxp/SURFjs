// import PairwiseSeamFinder from ;


//TODO
class GraphCutSeamFinder() {
    constructor() {
        this.dx = [];
        this.dy = [];
        this.costType = 0;
        this.terminalCost = 0;
        this.badRegionPenalty = 0;
        this.images = [];
        this.masks = [];
        this.corners = [];
        this.sizes = [];
    }

    genMatrix() {
    }

    sobel() {
    }

    normL2() {
    }

    find(src, corners, masks) {
        const numImages = src.length;
        const height = src[0].length;
        const width = src[0][0].length;
        for (let i = 0; i < numImages; i++) {
            let dx = [];
            let dy = [];
            this.sobel(src, dx, 1, 0);
            this.sobel(src, dy, 0, 1);
            this.dx[i] = this.genMatrix(height, width, 0);
            this.dy[i] = this.genMatrix(height, width, 0);
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    this.dx[i][y][x] = this.normL2(dx[y][x]);
                    this.dy[i][y][x] = this.normL2(dy[y][x]);
                }
            }
        }
        let pairwiseSeamFinder = new PairwiseSeamFinder();
        pairwiseSeamFinder.find(src, corners, masks);
    }

    setGraphWeightColor(img1, img2, mask1, mask2, graph) {
        const height = img1.length;
        const width = img1[0].length;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let v = graph.addVtx();
                graph.addTermWeights(v, mask1[y][x] ? this.terminalCost : 0.0, mask2[y][x] ? this.terminalCost : 0.0);
            }
        }
        let weightEps = 1.0;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let v = y * width + x;
                if (x < width - 1) {
                    let weight = this.normL2(img1[y][x], img2[y][x]) +
                        this.normL2(img1[y][x + 1], img2[y][x + 1]) + weightEps;
                    if (!mask1[y][x] || !mask1[y][x + 1] || 
                        !mask2[y][x] || !mask2[y][x + 1]) {
                        weight += this.badRegionPenalty;
                    }
                    graph.addEdges(v, v + width, weight, weight);
            }
        }
    }

    setGraphWeightColorGrad(img1, img2, dx1, dx2, dy1, dy2, mask1, mask2, graph) {
        const height = img1.length;
        const width = img1[0].length;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // TODO
            }
        }
    }

    findInPair(first, second, roi) {
        let img1 = this.images[first];
        let img2 = this.images[second];
        let mask1 = this.masks[first];
        let mask2 = this.masks[second];
        let dx1 = this.dx[first];
        let dx2 = this.dx[second];
        let dy1 = this.dy[first];
        let dy2 = this.dy[second];
        let tl1 = this.corners[first];
        let tl2 = this.corners[second];

        // TODO
    }


        

}
