function appendChild(parent, child) {
  if (child === null || child === undefined || child === false) {
    return;
  }

  if (Array.isArray(child)) {
    child.forEach((nestedChild) => appendChild(parent, nestedChild));
    return;
  }

  if (child instanceof Node) {
    parent.appendChild(child);
    return;
  }

  parent.appendChild(document.createTextNode(String(child)));
}

export function h(tagName, options = {}, children = []) {
  const element = document.createElement(tagName);
  const {
    attrs = {},
    className,
    dataset = {},
    html,
    on = {},
    text,
    ...props
  } = options;

  if (className) {
    element.className = className;
  }

  if (text !== undefined) {
    element.textContent = text;
  }

  if (html !== undefined) {
    element.innerHTML = html;
  }

  Object.entries(attrs).forEach(([name, value]) => {
    if (value === null || value === undefined || value === false) {
      return;
    }

    element.setAttribute(name, value === true ? '' : String(value));
  });

  Object.entries(dataset).forEach(([name, value]) => {
    if (value !== null && value !== undefined) {
      element.dataset[name] = String(value);
    }
  });

  Object.entries(on).forEach(([eventName, handler]) => {
    element.addEventListener(eventName, handler);
  });

  Object.entries(props).forEach(([name, value]) => {
    if (value === null || value === undefined || value === false) {
      return;
    }

    if (name in element) {
      element[name] = value;
      return;
    }

    element.setAttribute(name, value === true ? '' : String(value));
  });

  appendChild(element, children);

  return element;
}

export function mount(parent, child) {
  parent.replaceChildren(child);
}
