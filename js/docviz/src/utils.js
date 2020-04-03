
const _readFileInner = (filename, contentsCallback) => {
  const inner = (event) => {
    const contents = event.target.result;
    contentsCallback(contents);
  };
  let reader = new FileReader();
  reader.addEventListener('load', (event) => inner(event));
  return reader
};

export const readFileAsText = (filename, contentsCallback) => {
  const reader = _readFileInner(filename, contentsCallback);
  reader.readAsText(filename);
};

export const readFileAsDataURL = (filename, contentsCallback) => {
  const reader = _readFileInner(filename, contentsCallback);
  reader.readAsDataURL(filename);
};
