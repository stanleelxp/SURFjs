class PlaneWarper {
    constructor() {
        this.projector = new PlaneProjector();
    }

    warpPoint(pt, K, R, T) {
        this.projector.setCameraParams(K, R, T);
        let uv = {
            x: 0,
            y: 0
        };
        this.projector.mapForward(pt,x pt.y, uv.x, uv.y);
    }

    buildMaps(srcSize, K, R, T, xmap, ymap) {
        this.projector.setCameraParams(K, R, T);
        // TL: Top Left
        let dstTL = {
            x: 0,
            y: 0
        };
        let dstBR = {
            x: 0,
            y: 0
        };
        this.detectResultROI(srcSize, dstTL, dstBR);
        // let dstSize = [dstBR.x - dstTL.x + 1, dstBR.y - dstTL.y + 1];
        let dstSize = {
            x: dstBR.x - dstTL.x + 1,
            y: dstBR.y - dstTL.y + 1
        }

        xmap = this.genMatrix(dstSize.y, dstSize.x, 0);
        ymap = this.genMatrix(dstSize.y, dstSize.x, 0);
        let x = 0;
        let y = 0;
        for (let v = dstTL.y; v <= dstBR.y; v++) {
            for (let u = dstTL.x; u <= dstBR.x; u++) {
                this.projector.mapBackward(u, v, x, y);
                xmap[v - dstTL.y][u - dstTL.x] = x;
                ymap[v - dstTL.y][u - dstTL.x] = y;
            }
        }
        const rect = {
            tl: {
                x: dstTL.x,
                y: dstTL.y
            },
            height: dstBR.y - dstTL.y,
            width: dstBR.x - dstTL.x
        };
        return rect; 
    }


    warp(src, K, R, T, interpMode, borderMode, dst) {
        let uxmap = [];
        let uymap = [];
        let dstROI = this.buildMaps(src.size, K, R, T, uxmap, uymap);
        dst = this.genMatrix(dstROI.height, dstROI.width, 0);
        this.remap(src, dst, uxmap, uymap, interpMode, borderMode);
        return dstROI.tl;
    }

    warpROI(srcSize, K, R, T) {
        this.projector.setCameraParams(K, R, T);
        let dstTL = {
            x: 0,
            y: 0
        };
        let dstBR = {
            x: 0,
            y: 0
        };
        this.detectResultROI(srcSize, dstTL, dstBR);
        const rect = {
            tl: {
                x: dstTL.x,
                y: dstTL.y
            },
            height: dstBR.y + 1 - dstTL.y,
            width: dstBR.x + 1 - dstTL.x
        };
        return rect; 
    }

    detectResultROI(srcSize, dstTL, dstBR) {
        let TLU = Number.MAX_VALUE;
        let TLV = Number.MAX_VALUE;
        let BRU = -Number.MAX_VALUE;
        let BRV = -Number.MAX_VALUE;
        let u = 0;
        let v = 0;
        this.projector.mapForward(0, 0, u, v);
        TLU = Math.min(TLU, u);
        TLV = Math.min(TLV, v);
        BRU = Math.max(BRU, u);
        BRV = Math.max(BRV, v);
        this.projector.mapForward(0, srcSize.height - 1, u, v);
        TLU = Math.min(TLU, u);
        TLV = Math.min(TLV, v);
        BRU = Math.max(BRU, u);
        BRV = Math.max(BRV, v);
        this.projector.mapForward(srcSize.width - 1, srcSize.height - 1, u, v);
        TLU = Math.min(TLU, u);
        TLV = Math.min(TLV, v);
        BRU = Math.max(BRU, u);
        BRV = Math.max(BRV, v);

        dstTL.x = TLU;
        dstTL.y = TLV;
        dstBR.x = BRU;
        dstBR.y = BRV;
    }
}

