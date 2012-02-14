#ifdef LIN_MODE
#include "virtual_asm.h"
#include <sys/mman.h>
#include <unistd.h>
#include <mutex>

namespace assembler {

CodePage::CodePage(unsigned int Size, void* requestedStart) : used(0), final(false) {
	unsigned minPageSize = getMinimumPageSize();
	unsigned pages = Size / minPageSize;

	if(Size % minPageSize != 0)
		pages += 1;

	page = mmap(
		requestedStart,
		Size,
		PROT_READ | PROT_WRITE | PROT_EXEC,
		MAP_ANONYMOUS | MAP_PRIVATE,
		0,
		0);

	size = pages * minPageSize;
}

void CodePage::grab() {
	++references;
}

void CodePage::drop() {
	if(--references == 0)
		delete this;
}

CodePage::~CodePage() {
	munmap(page, size);
}

void CodePage::finalize() {
	mprotect(page, size, PROT_READ | PROT_EXEC);
	final = true;
}

unsigned int CodePage::getMinimumPageSize() {
	return getpagesize();
}

void CriticalSection::enter() {
	((std::mutex*)pLock)->lock();
}

void CriticalSection::leave() {
	((std::mutex*)pLock)->unlock();
}

CriticalSection::CriticalSection() {
	std::mutex* mutex = new std::mutex();
	pLock = mutex;
}
CriticalSection::~CriticalSection() {
	delete (std::mutex*)pLock;
}

};
#endif
