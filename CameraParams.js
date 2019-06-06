class CameraParams {
    constructor() {
        this.focal = 1;
        this.aspect = 1;
        this.ppx = 0;
        this.ppy = 0;
        this.R = this.genEye(3);
        this.T = this.genMatrix(3, 1, 0);
        this.projector = new PlaneProjector();
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

    genEye(n) {
        if (n === undefined || N < 1) {
            return [];
        } else {
            let mat = this.genMatrix(n, n, 0);
            for (let i = 0; i < n; i++) {
                mat[i][i] = 1;
            }
            return mat;
        }
    }

    K() {
        let K = this.genEye(3);
        k[0][0] = this.focal;
        k[0][2] = this.ppx;
        k[1][1] = this.focal * this.aspect;
        k[1][2] = this.ppy;
        return K;
    }
    update(options) {
        this.focal = options.focal || this.focal;
        this.aspect = options.aspect || this.aspect;
        this.ppx = options.ppx || this.ppx;
        this.ppy = options.ppy || this.ppy;
        this.R = options.R || this.R;
        this.T = options.T || this.T;
    }
}
