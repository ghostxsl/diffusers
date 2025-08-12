template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>JSON 图像 & 音乐展示（懒加载 + 强制刷新）</title>

  <!-- 强制浏览器每次都重新拉取 HTML / 资源 -->
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
  <meta http-equiv="Pragma" content="no-cache" />
  <meta http-equiv="Expires" content="0" />

  <style>
    body { margin: 20px; font-family: sans-serif; }

    /* 每条记录的外框 */
    .item-block {
      border: 2px solid #969696;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 32px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    /* 信息表格 */
    .info-table { border-collapse: collapse; }
    .info-table td { padding: 4px 8px; }
    .info-table td:first-child { font-weight: bold; }

    /* 图片容器 */
    .image-container {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .image-container img {
      max-width: 270px;
      max-height: 480px;
      width: auto;
      height: auto;
      object-fit: contain;
      background-color: #f0f0f0;
    }

    .image-diff-container {
      display: flex;
      overflow-x: auto;
      gap: 12px;
      padding-top: 8px;
    }

    /* 图片对比 */
    .diff-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      min-width: 270px;
    }

    .diff-item img {
      max-width: 270px;
      max-height: 480px;
      width: auto;
      height: auto;
      object-fit: contain;
      border-radius: 6px;
      background-color: #f0f0f0; /* 可选：防止背景透明难看 */
    }

    .diff-title {
      margin-top: 6px;
      font-size: 14px;
      color: #333;
      text-align: center;
      word-break: break-word;
    }
  </style>
</head>
<body>
  <div id="root"></div>

  <script>
    /*
      data 既可为单个对象，也可为对象数组 —— 生产环境只需替换此变量内容
    */
    var data = ___input_json_data___;

    /* ---------------- 兼容：若不是数组，则包裹成数组 ---------------- */
    if (!Array.isArray(data)) data = [data];

    var root = document.getElementById('root');

    /* -------------------------- DOM 渲染 -------------------------- */
    data.forEach(function(item){
      var block = document.createElement('div');
      block.className = 'item-block';

      /* 1) 信息表格 */
      var table = document.createElement('table');
      table.className = 'info-table';
      Object.keys(item).forEach(function(k){
        if (k.toLowerCase().includes('image') || k === 'MusicUrl') return;
        var tr = document.createElement('tr');
        var tdKey = document.createElement('td'); tdKey.textContent = k;
        var tdVal = document.createElement('td');
        var v = item[k];
        tdVal.textContent = (typeof v === 'object') ? JSON.stringify(v) : v;
        tr.appendChild(tdKey); tr.appendChild(tdVal); table.appendChild(tr);
      });
      block.appendChild(table);

      /* 2) 音乐播放器 (懒加载) */
      if (item.MusicUrl) {
        var audio = document.createElement('audio');
        audio.controls = true;
        audio.autoplay = false;
        // 随机参数避免缓存
        var musicUrl = item.MusicUrl + (item.MusicUrl.includes('?') ? '&' : '?') + 't=' + Date.now();
        audio.dataset.src = musicUrl;
        audio.preload = 'none';
        audio.className = 'lazy-media';
        block.appendChild(audio);
      }

      /* 5) 视频播放器 (懒加载) */
      if (item.VideoUrl) {
        var video = document.createElement('video');
        video.controls = true;
        video.width = 240;
        video.height = 480;
        video.autoplay = false;
        video.preload = 'none';
        video.className = 'lazy-media';
        video.dataset.src = item.VideoUrl + (item.VideoUrl.includes('?') ? '&' : '?') + 't=' + Date.now();
        block.appendChild(video);
      }


      /* 6) 子页面 iframe 和图片并排展示 */
      if (item.SubHTML || Array.isArray(item.ImageUrls)) {
        var mediaRow = document.createElement('div');
        mediaRow.style.display = 'flex';
        mediaRow.style.flexDirection = 'row';
        mediaRow.style.flexWrap = 'nowrap';
        mediaRow.style.alignItems = 'flex-start';
        mediaRow.style.gap = '16px';  // iframe 和图片之间间距

        // iframe 区域
        if (item.SubHTML) {
          var iframeWrapper = document.createElement('div');
          iframeWrapper.style.width = '280px';
          iframeWrapper.style.height = '500px';
          iframeWrapper.style.overflow = 'hidden';
          iframeWrapper.style.border = '1px solid #ccc';
          iframeWrapper.style.borderRadius = '0px';
          iframeWrapper.style.position = 'relative';
          iframeWrapper.style.flexShrink = '0';

          var iframe = document.createElement('iframe');
          iframe.width = 500;
          iframe.height = 500;
          iframe.style.border = 'none';
          iframe.style.position = 'absolute';
          iframe.style.left = '0px';
          iframe.style.top = '0px';
          // iframe.style.transform = `scale(0.9)`;
          iframe.style.transformOrigin = 'left bottom';
          iframe.className = 'lazy-media';
          iframe.dataset.src = item.SubHTML + (item.SubHTML.includes('?') ? '&' : '?') + 't=' + Date.now();

          iframeWrapper.appendChild(iframe);
          mediaRow.appendChild(iframeWrapper);
        }

        // 图片区域
        if (Array.isArray(item.ImageUrls)) {
          var imgWrap = document.createElement('div');
          imgWrap.className = 'image-container';
          item.ImageUrls.forEach(function(src){
            var img = document.createElement('img');
            img.loading = 'lazy';
            img.alt = 'image';
            img.dataset.src = src + (src.includes('?') ? '&' : '?') + 't=' + Date.now();
            img.className = 'lazy-media';
            imgWrap.appendChild(img);
          });
          mediaRow.appendChild(imgWrap);
        }

        block.appendChild(mediaRow);
      }

       /* 4) ImageDiff 横向图片列表 */
      if (Array.isArray(item.ImageDiff)) {
        var diffWrap = document.createElement('div');
        diffWrap.className = 'image-diff-container';

        item.ImageDiff.forEach(function(diffItem){
          if (!diffItem || typeof diffItem.imageUrl !== 'string') return;
          var diffItemBlock = document.createElement('div');
          diffItemBlock.className = 'diff-item';

          var img = document.createElement('img');
          img.loading = 'lazy';
          img.alt = diffItem.title || '';
          img.dataset.src = diffItem.imageUrl + (diffItem.imageUrl.includes('?') ? '&' : '?') + 't=' + Date.now();
          img.className = 'lazy-media';

          var title = document.createElement('div');
          title.className = 'diff-title';
          title.textContent = diffItem.title || '';

          diffItemBlock.appendChild(img);
          diffItemBlock.appendChild(title);
          diffWrap.appendChild(diffItemBlock);
        });

        block.appendChild(diffWrap);
      }


      root.appendChild(block);
    });

    /* ---------------------- 懒加载逻辑 (IntersectionObserver) ---------------------- */
    (function(){
      var lazyNodes = [].slice.call(document.querySelectorAll('.lazy-media'));
      if (!('IntersectionObserver' in window)) {
        // 不支持时立即加载
        lazyNodes.forEach(loadMedia);
        return;
      }

      var io = new IntersectionObserver(function(entries){
        entries.forEach(function(ent){
          if (ent.isIntersecting) {
            loadMedia(ent.target);
            io.unobserve(ent.target);
          }
        });
      }, { rootMargin: '200px 0px', threshold: 0.1 });

      lazyNodes.forEach(function(node){ io.observe(node); });

      /* 真正替换 src 的函数 */
      function loadMedia(el){
        var src = el.dataset.src;
        if (!src) return;
        el.src = src;
        if (el.tagName.toLowerCase() === 'audio') {
          el.load();
        }
        delete el.dataset.src;
      }
    })();

    /* ---------- 若页面从 BFCache 返回，强制刷新 (保证最新 HTML) ---------- */
    window.addEventListener('pageshow', function(evt){
      if (evt.persisted) location.reload();
    });
  </script>
</body>
</html>

"""
