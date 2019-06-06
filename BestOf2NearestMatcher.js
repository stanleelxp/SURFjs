class BestOf2NearestMatcher {
    constructor() {
        this.matchConf = 0;
        this.numMatchesThresh1 = 0;
        this.numMatchesThresh2 = 0;
    }

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

    findHomography() {
    }
    determinant() {
    }

    match(features1, features2, matchesInfo) {
        if (matchesInfo.matches.length < this.numMatchesThresh1) {
            return;
        }

        let srcPoints = this.genMatrix(matchesInfo.matches.length, 2, 0);
        let dstPoints = this.genMatrix(matchesInfo.matches.length, 2, 0);
        for (let i = 0; i < matchesInfo.matches.length; i++) {
            const m = matchesInfo.matches[i];
            let px = features1.keypoints[m.queryIdx].x;
            let py = features1.keypoints[m.queryIdx].y;
            px -= features1.width * 0.5;
            py -= features1.height * 0.5;
            srcPoints[i][0] = px;
            srcPoints[i][1] = py;

            px = features2.keypoints[m.queryIdx].x;
            py = features2.keypoints[m.queryIdx].y;
            px -= features2.width * 0.5;
            py -= features2.height * 0.5;
            dstPoints[i][0] = px;
            dstPoints[i][1] = py;
        }
        matchesInfo.H = this.findHomography(srcPoints, dstPoints, matchesInfo.inliersMask, 'RANSAC');
        if (!matchesInfo.H || Math.abs(this.determinant(matchesInfo.H)) < 1e-6) {
            return;
        }
        matchesInfo.numInliers = 0;
        for (let i = 0; i < matchesInfo.inliersMask.length; i++) {
            if (matchesInfo,inliersMask[i]) {
                matchesInfo.numInliers += 1;
            }
        }
        matchesInfo.confidece = matchesInfo.numInliers / (8 + 0.3 * matchesInfo.matches.length);
        matchesInfo.confidece = matchesInfo.confidence > 3 ? 0.0 : matchesInfo.confidence;
        if (matchesInfo.numInliers < numMatchesThresh2) {
            return;
        }

        // for inliers
        // TODO create
        // srcPoints.create(1, matchesInfo.numInliers);
        // dstPoints.create(1, matchesInfo.numInliers);
        srcPoints = this.genMatrix(matchesInfo.numInliers, 2, 0);
        dstPoints = this.genMatrix(matchesInfo.numInliers, 2, 0);
        let inlierIdx = 0;
        for (let i = 0; i < matchesInfo.matches.length; i++) {
            if (!matchesInfo.inliersMask[i]) {
                continue;
            }
            const m = machesInfo.matches[i];
            let px = features1.keypoints[m.queryIdx].x;
            let py = features1.keypoints[m.queryIdx].y;
            px -= features1.width * 0.5;
            py -= features1.height * 0.5;
            srcPoints[i][0] = px;
            srcPoints[i][1] = py;

            px = features2.keypoints[m.queryIdx].x;
            py = features2.keypoints[m.queryIdx].y;
            px -= features2.width * 0.5;
            py -= features2.height * 0.5;
            dstPoints[i][0] = px;
            dstPoints[i][1] = py;
            inlierIdx += 1;
        }
        matchesInfo.H = this.findHomography(srcPoints, destPoints, 'RANSAC');
    }

    operator(features, pairMatches, mask, rangeWidth) {
        const numImages = features.length;
        let mask_ = this.genMatrix(numImages, numImages, 1);
        let nearPairs = new Pair();
        for (let i = 0; i < numImages - 1; i++) {
            for (let j = i + 1; j < Math.min(numImages, i + rangeWidth); j++) {
                if (features[i].keypoints.length > 0 && features[j].keypoints.length > 0 && mask_[i][j]) {
                    nearPairs.push(this.makePair(i, j));
                }
            }
        }
        pairMatches.resize(numImages * numImages);
        let body = new MatchPairsBody(this, features, pairMatches, nearPairs);
        // TODO
        body(0, nearPairs.length);
    }

    collectGarbage() {
    }
}
