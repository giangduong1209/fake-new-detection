const headerBlock = document.querySelector(".header-block");
const headerUniversity = document.querySelector(".header-block__left h1 a");
const headerItems = Array.from(
  document.querySelectorAll(".header_block__right ul li a")
);

window.addEventListener("scroll", function () {
  if (document.documentElement.scrollTop > 20) {
    headerBlock.style.backgroundColor = "rgba(255, 255, 255, 1)";
    headerUniversity.style.color = "#ec1c24";
    headerItems.forEach((item) => {
      item.style.color = "#ec1c24";
    });
  } else {
    headerBlock.style.backgroundColor = "rgba(255, 255, 255, 0.15)";
    headerUniversity.style.color = "#fff";
    headerItems.forEach((item) => {
      item.classList.remove("active");
      item.style.color = "#fff";
    });
  }
});

headerItems.forEach((item) => {
  item.addEventListener("click", (e) => {
    const activeClass = document.querySelector(".active");
    if (activeClass) {
      activeClass.classList.remove("active");
    }
    e.target.classList.add("active");
  });
});
