function getAllImage() {
	cv['onRuntimeInitialized']=()=>{
		let pagepics = document.getElementsByTagName("img");
		let testImg = chrome.runtime.getURL("test_imgs/test_img1.jpg");
		console.log(testImg);

		for(i = 0; i < pagepics.length; i++){
			pagepics[i].src = testImg;
		}
	}
};


// 메인 이미지를 가져와서 opencv로 조작해보려고 합니다.
function getMainImage() {
	cv['onRuntimeInitialized']=()=>{
		
		// 메인 이미지를 가져오고 출력 확인
		let main_img = document.getElementById("img1");
		console.log(main_img);
		

		// 아래와 같은 코드를 실행하면 gray에 흑백이미지가 저장됩니다.
		// 아래 코드와 같은 흐름을 따라가면서 생기는 문제점을 적어두겠습니다.
		/*  
			let mat = cv.imread(imgElement);
			let gray = new cv.Mat();
			cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY, 0);
			cv.imshow('grayImg', gray);
		*/

		/*
			let mat = cv.imread(main_img); 
			이 코드를 실행하면 아래와 같은 오류가 뜹니다.
			Uncaught (in promise) DOMException: Failed to execute 'getImageData' on 'CanvasRenderingContext2D': The canvas has been tainted by cross-origin data.
			
		*/

		/*
			main_img.crossOrigin = "Anonymous";
			let mat = cv.imread(main_img); 
			그래서 이렇게 바꾸면 다른 오류가 뜹니다.
			Access to image at 'https://imgnews.pstatic.net/image/648/2022/04/27/0000007627_001_20220427174901864.jpg?type=w430' from origin 'https://n.news.naver.com' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
		*/
		

		// 그래도 cv2가 동작함은 확인할 수 있습니다.
		let gray = new cv.Mat();
		console.log(gray)


		/* 
			그 이후의 코드...
			cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY, 0);
			console.log("cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY, 0)", gray)
		*/
	}
};



// 여기서 함수만 바궈주세요.
document.addEventListener('DOMContentLoaded', getAllImage(), false);
