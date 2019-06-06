class FeaturesMatcher {
    genMatrix(height, width, val) {
        if (height === undefined || height < 1 || width < 1) {
            return [];
        } else if (width === undefined) {
            let arr1D = [];
            for (let i = 0; i < height; i++) {
                arr1D.push(val);
            }
            return arr1D;
        } else {
            let arr1D = [];
            let arr2D = [];
            for (let i = 0; i < width; i++) {
                arr1D.push(val);
            }
            for (let i = 0; i < height; i++) {
                arr2D.push(arr1D.slice());
            }
            return arr2D;
        }
    }

    makePair(a, b) {
    }

    operator (features, pairMatches, mask) {
        const numImages = features.length;
        const mask_ = this.genMatrix(numImages, numImages, val);
        const nearPairs = new Pair();
        for (let i = 0; i < numImages; i++) {
            for (let j = i - 1; j < numImages; j++) {
                if (features[i].keypoints.length > 0 && features[j].keypoints.length > 0 && mask_[i][j]) {
                    nearPairs.push(this.makePair(i, j));
                }
            }
        }

        pairMatches.clear();
        pairMatches.resize(numImages * numImages);
        let body = new MatchPairsBody(this, features, pairMatches, nearPairs);
        // TODO
        body(0, nearPairs.length);
    }
}
