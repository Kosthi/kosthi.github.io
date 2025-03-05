/* 离开当前页面时修改网页标题，回到当前页面时恢复原来标题 */
window.onload = function() {
    var OriginTitile = document.title;
    var titleTime;
    document.addEventListener('visibilitychange', function() {
        if(document.hidden) {
        $('[rel="icon"]').attr('href', "/failure.ico");
        $('[rel="shortcut icon"]').attr('href', "/failure.ico");
        document.title = '智慧树上智慧果';
        clearTimeout(titleTime);
        } else {
        $('[rel="icon"]').attr('href', "/favicon-32x32.png");
        $('[rel="shortcut icon"]').attr('href', "/favicon-32x32.png");
        document.title = '智慧树下你和我';
        titleTime = setTimeout(function() {
            document.title = OriginTitile;
        }, 2000);
        }
    });
}

class RuntimeManager {
    constructor() {
		this.element = document.getElementById('run-time');
		this.startTime = new Date(this.element.dataset.startTime);
		this.i18nText = this.element.dataset.i18nRuntime;
		this.init();
    }

    init() {
      	if (!this.element || isNaN(this.startTime)) {
        	console.error('Runtime element or start time invalid');
        	return;
      	}
      	this.update();
      	setInterval(() => this.update(), 1000);
    }
  
    formatTime(unit) {
		return unit.toString().padStart(2, '0');
    }

    update() {
      	const diff = Date.now() - this.startTime;
      	const days = Math.floor(diff / 86400000);
      	const hours = Math.floor((diff % 86400000) / 3600000);
      	const minutes = Math.floor((diff % 3600000) / 60000);
      	const seconds = Math.floor((diff % 60000) / 1000);

      	const output = this.i18nText
        	.replace(/%%DAYS%%/g, days)
        	.replace(/%%HOURS%%/g, this.formatTime(hours))
        	.replace(/%%MINUTES%%/g, this.formatTime(minutes))
        	.replace(/%%SECONDS%%/g, this.formatTime(seconds));
    
      	this.element.innerHTML = output;
    }
}

// 自动初始化
document.addEventListener('DOMContentLoaded', () => new RuntimeManager());
